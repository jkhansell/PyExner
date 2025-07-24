from mpi4py import MPI

from typing import NamedTuple

class Neighbors(NamedTuple):
    north: int
    south: int
    east: int
    west: int

class Parallel:
    def __init__(self, params):
        
        self.parNx = params["parNx"]
        self.parNy = params["parNy"]

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        assert self.size == self.parNx * self.parNy, "Domain decomposition not possible nranks != parNx * parNy"

        # Define Cartesian communicator
        self.dims = (self.parNy, self.parNx)
        self.periods = [False, False]  # Non-periodic boundaries
        self.reorder = True
        self.cart_comm = self.comm.Create_cart(self.dims, periods=self.periods, reorder=self.reorder)

        self.coords = self.cart_comm.Get_coords(self.rank)

        nbr_west, nbr_east = self.cart_comm.Shift(1, 1)
        nbr_south, nbr_north = self.cart_comm.Shift(0, 1)

        self.neighbors = {
            "north" : nbr_north,
            "south" : nbr_south,
            "east" : nbr_east,
            "west" : nbr_west
        }
