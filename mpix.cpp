/*!
 * \file mpix.cpp
 * \author Han Luo (luohancfd AT github)
 * \brief Wrapper of MPI functions
 * \version 0.1
 * \date 2020-12-22
 * \copyright GPL-3.0
 *
 * @copyright Copyright (c) 2020
 *
 */

#include "mpix.hpp"

#include <sys/stat.h>

#include <climits>
#include <iostream>
#include <type_traits>

#include "traits.hpp"

int MPIX::Rank = 0;
int MPIX::localRank = 0;
int MPIX::Size = 1;
bool MPIX::isWorldSplitted = false;
MPIX::Comm MPIX::worldComm = MPI_COMM_WORLD;
MPIX::Comm MPIX::localComm = MPI_COMM_WORLD;

int MPIX::MinRankError = INT_MAX; /*!< \brief Minimum rank with error */
bool MPIX::winMinRankErrorInUse = false;
MPIX::Win MPIX::winMinRankError =
    0; /*!< \brief Window for communication of error */

/* ---------------------------------------------------------------------- */

int MPIX::GetRank() { return Rank; }

int MPIX::GetSize() { return Size; }

bool MPIX::isMaster() { return Rank == MPI_MASTER_PROCESS; }

MPIX::Comm MPIX::GetWorldComm() { return worldComm; }

MPIX::Comm MPIX::GetLocalComm() {
  if (isWorldSplitted) {
    std::cout << "World is splitted" << std::endl;
    Abort(1);
  }
  return localComm;
}

void MPIX::SplitComm(int color, MPIX::Comm comm) {
  MPI_Comm_split(comm, color, Rank, &localComm);
}

void MPIX::SetWorldComm(MPIX::Comm newComm) {
  worldComm = newComm;
  MPI_Comm_rank(worldComm, &Rank);
  MPI_Comm_size(worldComm, &Size);

  if (winMinRankErrorInUse) MPI_Win_free(&winMinRankError);
  MinRankError = Size;
  MPI_Win_create(&MinRankError, sizeof(int), sizeof(int), MPI_INFO_NULL,
                 worldComm, &winMinRankError);
  winMinRankErrorInUse = true;
}

void MPIX::Error(std::string ErrorMsg, std::string FunctionName) {
  /* Set MinRankError to Rank, as the error message is called on this rank. */
  MinRankError = Rank;
  int flag = 0;

  /* Find out whether the error call is collective via MPI_Ibarrier. */
  Request barrierRequest;
  Ibarrier(worldComm, &barrierRequest);

  /* Try to complete the non-blocking barrier call for a second. */
  double startTime = Wtime();
  while (true) {
    MPI_Test(&barrierRequest, &flag, MPI_STATUS_IGNORE);
    if (flag) break;

    double currentTime = Wtime();
    if (currentTime > startTime + 1.0) break;
  }

#ifndef MPI_STUBS
  if (flag) {
    /* The barrier is completed and hence the error call is collective.
       Set MinRankError to 0. */
    MinRankError = 0;
  } else {
    /* The error call is not collective and the minimum rank must be
       determined by one sided communication. Loop over the lower numbered
       ranks to check if they participate in the error message. */
    /* In human's words: only certain processes call this function */
    for (int i = 0; i < Rank; ++i) {
      int MinRankErrorOther;
      MPI_Win_lock(MPI_LOCK_SHARED, i, 0, winMinRankError);
      MPI_Get(&MinRankErrorOther, 1, MPI_INT, i, 0, 1, MPI_INT,
              winMinRankError);
      MPI_Win_unlock(i, winMinRankError);

      if (MinRankErrorOther < MinRankError) {
        MinRankError = MinRankErrorOther;
        break;
      }
    }
  }
#else
  MinRankError = 0;
#endif

  /* Check if this rank must write the error message and do so. */
  if (Rank == MinRankError) {
    std::cout << std::endl << std::endl;
    std::cout << "Error in \"" << FunctionName << "\": " << std::endl;
    std::cout << "-------------------------------------------------------------"
                 "------------"
              << std::endl;
    std::cout << ErrorMsg << std::endl;
    std::cout << "------------------------------ Error Exit "
                 "-------------------------------"
              << std::endl;
    std::cout << std::endl << std::endl;
  }
  Abort(EXIT_FAILURE);
}

void MPIX::Init(int &argc, char **&argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(worldComm, &Rank);
  MPI_Comm_size(worldComm, &Size);

  MinRankError = Size;
  MPI_Win_create(&MinRankError, sizeof(int), sizeof(int), MPI_INFO_NULL,
                 worldComm, &winMinRankError);
  winMinRankErrorInUse = true;
}

void MPIX::Finalize() {
  if (winMinRankErrorInUse) MPI_Win_free(&winMinRankError);
  MPI_Finalize();
}

void MPIX::Barrier(Comm comm) { MPI_Barrier(comm); }

void MPIX::Abort(int error, MPIX::Comm comm) { MPI_Abort(comm, error); }



void MPIX::Get_count(Status *status, Datatype datatype, int *count) {
  MPI_Get_count(status, datatype, count);
}

void MPIX::Isend(void *buf, int count, Datatype datatype, Request *request,
                 int dest, int tag, MPIX::Comm comm) {
  MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}
void MPIX::Isend(void *buf, int count, Datatype datatype, int dest, int tag,
                 MPIX::Comm comm, Request *request) {
  MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

void MPIX::Irecv(void *buf, int count, Datatype datatype, Request *request,
                 int src, int tag, MPIX::Comm comm) {
  MPI_Irecv(buf, count, datatype, src, tag, comm, request);
}
void MPIX::Irecv(void *buf, int count, Datatype datatype, int src, int tag,
                 MPIX::Comm comm, Request *request) {
  MPI_Irecv(buf, count, datatype, src, tag, comm, request);
}

void MPIX::Ibarrier(Comm comm, Request *request) {
  MPI_Ibarrier(comm, request);
}
void MPIX::Ibarrier(Request *request) { MPI_Ibarrier(worldComm, request); }

void MPIX::Wait(Request *request, Status *status) { MPI_Wait(request, status); }

void MPIX::Waitall(int nrequests, Request *request, Status *status) {
  MPI_Waitall(nrequests, request, status);
}

void MPIX::Waitany(int nrequests, Request *request, int *index,
                   Status *status) {
  MPI_Waitany(nrequests, request, index, status);
}
void MPIX::Waitany(int nrequests, Request *request, int &index,
                   Status *status) {
  MPI_Waitany(nrequests, request, &index, status);
}

void MPIX::Probe(int source, int tag, MPIX::Comm comm, Status *status) {
  MPI_Probe(source, tag, comm, status);
}

void MPIX::Bcast(void *buf, int count, Datatype datatype, int root,
                 MPIX::Comm comm) {
  MPI_Bcast(buf, count, datatype, root, comm);
}

void MPIX::Bcast(std::string &str, int root, MPIX::Comm comm) {
  unsigned int len = str.length();
  Bcast(&len, 1, MPI_UNSIGNED, root, comm);
  if (len > 0) {
    char *buffer;
    if (Rank == root) {
      buffer = const_cast<char *>(str.data());
    } else {
      buffer = new char[len];
    }
    Bcast(buffer, len, MPI_CHAR, root, comm);
    if (Rank != root) {
      str.assign(buffer, len);
      delete[] buffer;
    }
  } else {
    if (Rank != root) str.erase();
  }
}

void MPIX::Bsend(void *buf, int count, Datatype datatype, int dest, int tag,
                 MPIX::Comm comm) {
  MPI_Bsend(buf, count, datatype, dest, tag, comm);
}

void MPIX::Buffer_attach(void *buffer, int size) {
  MPI_Buffer_attach(buffer, size);
}

void MPIX::Buffer_detach(void *buffer, int *size) {
  MPI_Buffer_detach(buffer, size);
}

void MPIX::Reduce(void *sendbuf, void *recvbuf, int count, Datatype datatype,
                  Op op, int root, MPIX::Comm comm) {
  MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm);
}

void MPIX::Reduce(void *sendbuf, int count, Datatype datatype, Op op, int root,
                  MPIX::Comm comm) {
  MPI_Reduce(MPI_IN_PLACE, sendbuf, count, datatype, op, root, comm);
}

void MPIX::Allreduce(void *sendbuf, void *recvbuf, int count, Datatype datatype,
                     Op op, MPIX::Comm comm) {
  MPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

void MPIX::Allreduce(void *sendbuf, int count, Datatype datatype, Op op,
                     MPIX::Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, sendbuf, count, datatype, op, comm);
}

void MPIX::Gather(void *sendbuf, int sendcnt, Datatype sendtype, void *recvbuf,
                  int recvcnt, Datatype recvtype, int root, MPIX::Comm comm) {
  MPI_Gather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root,
             comm);
}

void MPIX::Scatter(void *sendbuf, int sendcnt, Datatype sendtype, void *recvbuf,
                   int recvcnt, Datatype recvtype, int root, MPIX::Comm comm) {
  MPI_Scatter(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, root,
              comm);
}

void MPIX::Allgather(void *sendbuf, int sendcnt, Datatype sendtype,
                     void *recvbuf, int recvcnt, Datatype recvtype,
                     MPIX::Comm comm) {
  MPI_Allgather(sendbuf, sendcnt, sendtype, recvbuf, recvcnt, recvtype, comm);
}

void MPIX::Allgatherv(void *sendbuf, int sendcount, Datatype sendtype,
                      void *recvbuf, int *recvcounts, int *displs,
                      Datatype recvtype, MPIX::Comm comm) {
  MPI_Allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs,
                 recvtype, comm);
}

void MPIX::Alltoall(void *sendbuf, int sendcount, Datatype sendtype,
                    void *recvbuf, int recvcount, Datatype recvtype,
                    MPIX::Comm comm) {
  MPI_Alltoall(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype,
               comm);
}

void MPIX::Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                     Datatype sendtype, void *recvbuf, int *recvcounts,
                     int *recvdispls, Datatype recvtype, MPIX::Comm comm) {
  MPI_Alltoallv(sendbuf, sendcounts, sdispls, sendtype, recvbuf, recvcounts,
                recvdispls, recvtype, comm);
}

void MPIX::Sendrecv(void *sendbuf, int sendcnt, Datatype sendtype, int dest,
                    int sendtag, void *recvbuf, int recvcnt, Datatype recvtype,
                    int source, int recvtag, MPIX::Comm comm, Status *status) {
  MPI_Sendrecv(sendbuf, sendcnt, sendtype, dest, sendtag, recvbuf, recvcnt,
               recvtype, source, recvtag, comm, status);
}

void MPIX::Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts,
                          Datatype datatype, Op op, MPIX::Comm comm) {
  MPI_Reduce_scatter(sendbuf, recvbuf, recvcounts, datatype, op, comm);
}

double MPIX::Wtime(void) { return MPI_Wtime(); }

int MPIX::split_job(int num_job) {
  int job_per_node = num_job / Size;
  int remainder = num_job % Size;

  if (Rank < remainder)
    return job_per_node + 1;
  else
    return job_per_node;
}

std::vector<int> MPIX::split_job(const std::vector<int> &num_job) {
  std::vector<int> new_jobs;
  new_jobs.reserve(num_job.size());
  for (auto i : num_job) {
    new_jobs.emplace(new_jobs.end(), split_job(i));
  }
  return new_jobs;
}

void MPIX::Consecutive_exec(std::function<void()> f, Comm comm) {
  int buf = Rank;
  if (Rank != 0) Recv(buf, Rank - 1);
  f();
  if (Rank < Size - 1) Send(buf, Rank + 1);
  Barrier(comm);
}

std::string MPIX::read_ascii_file(const char file_path[]) {
  long file_size = -1;
  if (isMaster()) {
    struct stat stat_buf;
    int rc = stat(file_path, &stat_buf);
    file_size = rc == 0 ? stat_buf.st_size : -1;
    file_size /= sizeof(char);
  }
  MPIX::Bcast(file_size);
  if (file_size < 0) {
    return "";
  }
  char *buf = new char[file_size];
  MPI_File fh;

  MPI_File_open(MPIX::GetWorldComm(), file_path, MPI_MODE_RDONLY, MPI_INFO_NULL,
                &fh);
  MPI_File_read_all(fh, buf, file_size, MPI_CHAR, MPI_STATUS_IGNORE);

  std::string content(buf, file_size);
  delete[] buf;
  MPIX::Barrier();
  return content;
}