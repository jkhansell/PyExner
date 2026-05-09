from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

left  = rank - 1 if rank > 0 else MPI.PROC_NULL
right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

send_right = np.array([rank], dtype=np.int32)
recv_left  = np.array([-1], dtype=np.int32)

send_left  = np.array([rank], dtype=np.int32)
recv_right = np.array([-1], dtype=np.int32)

# ==========================================
# Send right, receive from left
# ==========================================

comm.Sendrecv(
    sendbuf=send_right,
    dest=right,
    recvbuf=recv_left,
    source=left,
)

# ==========================================
# Send left, receive from right
# ==========================================

comm.Sendrecv(
    sendbuf=send_left,
    dest=left,
    recvbuf=recv_right,
    source=right,
)

print(
    f"Rank {rank}: "
    f"recv_left={recv_left[0]} "
    f"recv_right={recv_right[0]}"
)