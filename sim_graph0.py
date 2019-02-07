#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python libraries
import numpy as np
import pandas as pd
import copy
import logging
import os

from time import time

from scipy.sparse import csr_matrix, identity, diags, issparse
# This is not being used, because it is quit slow
# from scipy.stats import entropy

from sklearn.neighbors import radius_neighbors_graph

EPS = np.finfo(float).tiny


class SimGraph(object):

    """
    Generic class to generate similarity graphs from data
    """

    def __init__(self, Tg,  out_path=N one, label="dg"):

        """
        Stores the main attributes of a datagraph object and loads the graph
        data as a list of node attributes from a database
        """

        # #########################################
        # Internal name assigned to the graph object
        self.label = label

        # ###############
        # Graph variables
        self.Tg = Tg                       # Data matrix
        self.n_nodes, self.dim = Tg.shape  # Data dimensions

        # ###############
        # Other variables

        # Path for the output files
        self.out_path = out_path

        #
        self.edgeT_id = None   # List of edges, as pairs (i, j) of indices.
        return

    def computeGraph(self, R=None, similarity='He', g=1, th_gauss=0.1):
        """
        Computes a sparse graph for the self graph structure.
        The self graph must containg a T-matrix, self.T

        Inputs:
            :self.T:   Data matrix
            :R: Radius. Edges link all data pairs at distance lower than R
                This is to forze a sparse graph.
            :similarity: Similarity measure used to compute affinity matrix
                Available options are:
                    'l1'     :1 minus l1 distance
                    'He'     :1 minus squared Hellinger's distance (JS)
                              (sklearn-based implementation)
                    'Gauss'  :An exponential function of the squared l2
                              distance
            :g: Exponent for the affinity mapping (not used for 'Gauss')
            :th_gauss:  Similarity threshold All similarity values below this
                threshold are set to zero. This is only for the gauss method,
                the rest of them compute the threshold automatically from R).

        Returns:
            :self.edgeT_id:  List of edges, as pairs (i, j) of indices
            :self.affinityT: List of affinity values for each pair in edgeT_id
            :self.df_edges:  Pandas dataframe with one row per edge and columns
                'Source', 'Target' and 'Weihgt'. The weight is equal to the
                (mapped) affinity value
        """

        logging.info(f"-- Computing graph with {self.n_nodes} nodes")
        logging.info(f"-- Similarity measure: {similarity}")

        # #########################
        # Computing Distance Matrix

        # This is just to abbreviate
        Tg = self.Tg

        # Select Distance measure for radius_neighbor_graph
        if similarity in ['Gauss', 'He']:
            d = 'l2'     # Note: l2 seems equivalent to minkowski (p=2)
        elif similarity in ['l1']:
            d = 'l1'     # Note: l1 seems equivalent to manhattan
        else:
            logging.error("computeTsubGraph ERROR: Unknown similarity measure")
            exit()

        # Select secondary radius
        R0 = R

        # Compute the connectivity graph of all pair of nodes at distence
        # below R0
        # IMPORTANT: Note that, despite radius_neighbors_graph has an option
        # 'distance' that returns the distance values, it cannot be used in
        # any case because the distance matrix does not distinghish between
        # nodes at distance > R0 and nodes at distance = 0
        t0 = time()
        logging.info(f'-- -- Computing neighbors_graph ...')
        if similarity in ['He']:
            # We must compute the connectivity graph because module
            # radius_neighbors_graph looses edges between nodes at zero
            # distance
            D = radius_neighbors_graph(np.sqrt(Tg), radius=R0,
                                       mode='connectivity', metric=d)
        elif similarity in ['l1', 'Gauss']:
            D = radius_neighbors_graph(Tg, radius=R0, mode='connectivity',
                                       metric=d)

        logging.info(f'       in {time()-t0} seconds')

        # ##############################################
        # From distance matrix to list of weighted edges

        # Compute lists with origin, destination and value for all edges in
        # the graph affinity matrix.
        orig_id, dest_id = D.nonzero()

        # Since the graph is undirected, we select ordered pairs orig_id,
        # dest_id only
        self.edgeT_id = list(filter(lambda i: i[0] < i[1],
                             zip(orig_id, dest_id)))

        # ####################
        # Computing Affinities

        logging.info(f"-- -- Computing affinities for {len(self.edgeT_id)}" +
                     " edges ...",)
        t0 = time()

        if similarity == 'He':
            # A new self.edgeT_id is returned because the function filters out
            # affinity values below th.
            self.edgeT_id, self.affinityT = self.he_affinity(Tg, R, g)

        elif similarity == 'l1':
            self.edgeT_id, self.affinityT = self.l1_affinity(Tg, R, g)

        elif similarity == 'Gauss':
            self.edgeT_id, self.affinityT = self.l2_affinity(Tg, R, th_gauss)
        else:
            logging.error("computeTsubGraph ERROR: Unknown similarity measure")

        logging.info(f"      reduced to {len(self.edgeT_id)} edges")
        logging.info(f'      Computed in {time()-t0} seconds')

        logging.info(("-- -- Graph generated with {0} nodes and {1} " +
                      "edges").format(self.n_nodes, len(self.edgeT_id)))

        return

    def he_affinity(self, Tg, R=1, g=1, blocksize=1_000_000):
        """ Compute all Hellinger's affinities between all nodes in the
            graph based on the node attribute vectos
            It assumes that all attribute vectors are normalized to sum up to 1
            Attribute matrix Tg can be sparse

            Args:

                Tg  :Matrix of probabilistic attribute vectors
                R   :Maximum JS distance. Edges at higher distance are removed
                g   :Exponent for the finnal affinity mapping
        """

        # ################################
        # Compute affinities for all edges

        # This is just to make sure that blocksize is an integer, to avoid an
        # execution error when used as an array index
        blocksize = int(blocksize)

        # I take the square root here. This is inefficient if Tg has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).
        X = np.sqrt(Tg)

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when Tg is sparse.
        d2_he = []
        for i in range(0, len(self.edgeT_id), blocksize):
            edge_ids = self.edgeT_id[i: i + blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)

            if issparse(Tg):
                P = X[list(i0)].toarray()
                Q = X[list(i1)].toarray()
            else:
                P = X[list(i0)]
                Q = X[list(i1)]

            # Squared Hellinger's distance
            # The maximum is used here just to avoid 2-2s<0 due to
            # precision errors
            s = np.sum(P * Q, axis=1)
            d2_he += list(np.maximum(2 - 2 * s, 0))

        # #########
        # Filtering

        # Filter out edges with JS distance above R (divergence above R**2).
        edge_id = [z[0] for z in zip(self.edgeT_id, d2_he) if z[1] < R**2]

        # ####################
        # Computing affinities

        # The final affinity values are computed using a transformation the
        # states a minimum affinity value equal to zero
        affinityT = [(1 - z/R**2)**g for z in d2_he if z < R**2]

        return edge_id, affinityT

    def l1_affinity(self, Tg, R=1, g=1, blocksize=1_000_000):
        """ Compute all l1's affinities between all nodes in the graph based on
            the node attribute vectors
            It assumes that all attribute vectors are normalized to sum up to 1
            Attribute matrix Tg can be sparse

            Args:

                Tg  :Matrix of probabilistic attribute vectors
                R   :Maximum JS distance. Edges at higher distance are removed
                g   :Exponent for the finnal affinity mapping
        """

        # ################################
        # Compute affinities for all edges

        # This is just to make sure that blocksize is an integer, to avoid an
        # execution error when used as an array index
        blocksize = int(blocksize)

        # I take the square root here. This is inefficient if Tg has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when Tg is sparse.
        d_l1 = []
        for i in range(0, len(self.edgeT_id), blocksize):
            edge_ids = self.edgeT_id[i: i + blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)
            if issparse(Tg):
                P = Tg[list(i0)].toarray()
                Q = Tg[list(i1)].toarray()
            else:
                P = Tg[list(i0)]
                Q = Tg[list(i1)]

            # l1 distance
            d_l1 += list(np.sum(np.abs(P - Q), axis=1))

        # #########
        # Filtering

        # Filter out edges with JS distance above R (divergence above R**2).
        edge_id = [z[0] for z in zip(self.edgeT_id, d_l1) if z[1] < R**2]

        # ####################
        # Computing affinities

        # The final affinity values are computed using a transformation the
        # states a minimum affinity value equal to zero
        affinityT = [(1 - z / R)**g for z in d_l1 if z < R]

        return edge_id, affinityT

    def l2_affinity(self, Tg, R=1, th_gauss=0.1, blocksize=1_000_000):
        """ Compute all l2's affinities between all nodes in the graph based on
            the node attribute vectors
            It assumes that all attribute vectors are normalized to sum up to 1
            Attribute matrix Tg can be sparse

            Args:

                Tg  :Matrix of probabilistic attribute vectors
                R   :Maximum JS distance. Edges at higher distance are removed
                :th_gauss:  Similarity threshold All similarity values below
                     this threshold are set to zero. This is only for the gauss
                     method, the rest of them compute the threshold
                     automatically from R).
        """

        # ################################
        # Compute affinities for all edges

        # This is just to make sure that blocksize is an integer, to avoid an
        # execution error when used as an array index
        blocksize = int(blocksize)

        # I take the square root here. This is inefficient if Tg has many
        # rows and just af few edges will be computed. However, we can
        # expect the opposite (the list of edges involves the most of the
        #  nodes).

        # Divergences are compute by blocks. This is much faster than a
        # row-by-row computation, specially when Tg is sparse.
        d_l2 = []
        for i in range(0, len(self.edgeT_id), blocksize):
            edge_ids = self.edgeT_id[i: i + blocksize]

            # Take the (matrix) of origin and destination attribute vectors
            i0, i1 = zip(*edge_ids)
            if issparse(Tg):
                P = Tg[list(i0)].toarray()
                Q = Tg[list(i1)].toarray()
            else:
                P = Tg[list(i0)]
                Q = Tg[list(i1)]

            # l1 distance
            d_l2 += list(np.sum((P - Q)**2, axis=1))

        # #########
        # Filtering

        # Filter out edges with JS distance above R (divergence above R**2).
        edge_id = [z[0] for z in zip(self.edgeT_id, d_l2) if z[1] < R**2]

        # ####################
        # Computing affinities

        # The value of gamma to get min edge weight th_gauss at distance R
        gamma = - np.log(th_gauss) / R**2
        # Nonzero affinity values
        affinityT = [np.exp(-gamma * z) for z in d_l2 if z < R**2]

        return edge_id, affinityT
