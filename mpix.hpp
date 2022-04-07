/*!
 * \file mpix.hpp
 * \author Han Luo (luohancfd AT github)
 * \brief Wrapper of MPI functions
 * \version 0.1
 * \date 2020-12-22
 * \copyright GPL-3.0
 *
 * @copyright Copyright (c) 2020
 *
 */
#include <climits>
#include <cstdint>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#include "mpi.h"
#include "traits.hpp"

#pragma once
#ifndef MPIX_HPP
#define MPIX_HPP

#ifndef MPI_DEFAULT_SEND_TAG
#define MPI_DEFAULT_SEND_TAG 0
#endif

#ifndef MPI_MASTER_PROCESS
#define MPI_MASTER_PROCESS 0
#endif

#if SIZE_MAX == UCHAR_MAX
#define MPI_SIZE_T MPI_UNSIGNED_CHAR
#elif SIZE_MAX == USHRT_MAX
#define MPI_SIZE_T MPI_UNSIGNED_SHORT
#elif SIZE_MAX == UINT_MAX
#define MPI_SIZE_T MPI_UNSIGNED
#elif SIZE_MAX == ULONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG
#elif SIZE_MAX == ULLONG_MAX
#define MPI_SIZE_T MPI_UNSIGNED_LONG_LONG
#else
#error "what is happening here?"
#endif

// clang-format off
template <typename T2, typename T = typename std::remove_const<T2>::type>
struct MPIX_TYPE {
  constexpr static int type = MPIX_TYPE<T>::type;
};

template <> struct MPIX_TYPE<bool> { constexpr static int type = MPI_INT; };

template <> struct MPIX_TYPE<char> { constexpr static int type = MPI_CHAR; };

template <> struct MPIX_TYPE<short> { constexpr static int type = MPI_SHORT; };

template <> struct MPIX_TYPE<int> { constexpr static int type = MPI_INT; };

template <> struct MPIX_TYPE<long> { constexpr static int type = MPI_LONG; };

template <> struct MPIX_TYPE<long long> { constexpr static int type = MPI_LONG_LONG; };

template <> struct MPIX_TYPE<unsigned short> { constexpr static int type = MPI_UNSIGNED_SHORT; };

template <> struct MPIX_TYPE<unsigned int> { constexpr static int type = MPI_UNSIGNED; };

template <> struct MPIX_TYPE<unsigned long> { constexpr static int type = MPI_UNSIGNED_LONG; };

template <> struct MPIX_TYPE<unsigned long long> { constexpr static int type = MPI_UNSIGNED_LONG_LONG; };

template <> struct MPIX_TYPE<float> { constexpr static int type = MPI_FLOAT; };

template <> struct MPIX_TYPE<double> { constexpr static int type = MPI_DOUBLE; };

template <> struct MPIX_TYPE<wchar_t> { constexpr static int type = MPI_WCHAR; };
// clang-format on

/**
 * @brief Wrapper of mpi functions, use MPIX instead of MPI to avoid
 * conflicition with IntelMPI
 *
 */
class MPIX {
public:
  typedef MPI_Request Request;
  typedef MPI_Status Status;
  typedef MPI_Datatype Datatype;
  typedef MPI_Op Op;
  typedef MPI_Comm Comm;
  typedef MPI_Win Win;

protected:
  static bool isWorldSplitted;
  static Comm worldComm, localComm;
  static int MinRankError; /*!< \brief Minimum rank with error */
  static bool winMinRankErrorInUse;
  static Win winMinRankError; /*!< \brief Window for communication of error */

public:
  static int Rank, localRank, Size;

public:
  static int GetRank();

  static int GetSize();

  static bool isMaster();

  static Comm GetWorldComm();

  static Comm GetLocalComm();

  static void SplitComm(int color, Comm comm = worldComm);

  static void SetWorldComm(Comm NewComm);

  static void Error(std::string ErrorMsg, std::string FunctionName);

  static void Init(int &argc, char **&argv);

  static void Finalize();

  static void Barrier(Comm comm = worldComm);

  /*!
   *   \attention The order of argument is different from MPI_abort
   */
  static void Abort(int error, Comm comm = worldComm);

  /* ---------------------------------------------------------------------- */

  /**
   * @brief Send one value
   *
   * @tparam T
   * @tparam std::enable_if<is_numeric<T>::value, bool>::type
   * @param value
   * @param dest
   * @param tag
   * @param comm
   */
  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Send(const T &value, int dest, int tag = MPI_DEFAULT_SEND_TAG,
                   Comm comm = worldComm) {
    MPI_Send(&value, 1, MPIX_TYPE<T>::type, dest, tag, comm);
  }

  // /**
  //  * @brief Send an array, array size must be less than INT_MAX
  //  *
  //  * @tparam T
  //  * @tparam std::enable_if<is_numeric<T>::value, bool>::type
  //  * @param buf
  //  * @param count
  //  * @param dest
  //  * @param tag
  //  * @param comm
  //  */
  // template <typename T,
  //           typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  // static void Send(T *buf, int count, int dest, int tag =
  // MPI_DEFAULT_SEND_TAG,
  //                  Comm comm = worldComm) {
  //   MPI_Send(buf, count, MPIX_TYPE<T>::type, dest, tag, comm);
  // }

  /**
   * @brief Send an array, array size must be less than ULONG_MAX but can be
   *        larger than INT_MAX
   *
   * @tparam T
   * @tparam std::enable_if<is_numeric<T>::value, bool>::type
   * @param buf
   * @param count
   * @param dest
   * @param tag
   * @param comm
   */
  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Send(T *buf, unsigned long count, int dest,
                   int tag = MPI_DEFAULT_SEND_TAG, Comm comm = worldComm) {
    while (count > INT_MAX) {
      MPI_Send(buf, INT_MAX, MPIX_TYPE<T>::type, dest, tag, comm);
      count -= INT_MAX;
      buf += INT_MAX;
    }
    MPI_Send(buf, (int)count, MPIX_TYPE<T>::type, dest, tag, comm);
  }

  /**
   * @brief Send a general data type following MPI standard
   *
   * @param buf
   * @param count
   * @param datatype
   * @param dest
   * @param tag
   * @param comm
   */
  static void Send(void *buf, int count, Datatype datatype, int dest,
                   int tag = MPI_DEFAULT_SEND_TAG, Comm comm = worldComm) {
    MPI_Send(buf, count, datatype, dest, tag, comm);
  }

  /**
   * @brief Send a std::string
   *
   * @param str
   * @param dest
   * @param tag
   * @param comm
   */
  static void Send(std::string const &str, int dest,
                   int tag = MPI_DEFAULT_SEND_TAG, Comm comm = worldComm) {
    unsigned long len = str.size();
    MPIX::Send(len, dest, tag, comm);
    if (len != 0) {
      MPIX::Send<>(str.data(), len, dest, tag, comm);
    }
  }

  /**
   * @brief Send a std::vector containing fundamental data type
   *
   * @tparam T
   * @tparam A
   * @tparam std::enable_if<std::is_fundamental<T>::value, bool>::type
   * @param vec
   * @param dest
   * @param tag
   * @param comm
   */
  template <
      typename T, typename A,
      typename std::enable_if<std::is_fundamental<T>::value, bool>::type = true>
  static void Send(std::vector<T, A> const &vec, int dest,
                   int tag = MPI_DEFAULT_SEND_TAG, Comm comm = worldComm) {
    unsigned long len = vec.size();
    MPIX::Send(len, dest, tag, comm);
    if (len != 0) {
      MPIX::Send<>(vec.data(), len, dest, tag, comm);
    }
  }

  /* ---------------------------------------------------------------------- */

  /**
   * @brief Receive a value from a processor
   *
   * @tparam T
   * @tparam std::enable_if<is_numeric<T>::value, bool>::type
   * @param value
   * @param src
   * @param tag
   * @param comm
   * @param status
   */
  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Recv(T &value, int src = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG,
                   Comm comm = worldComm, Status *status = MPI_STATUS_IGNORE) {
    MPI_Recv(&value, 1, MPIX_TYPE<T>::type, src, tag, comm, status);
  }

  /**
   * @brief Receive an array, array size must be less than ULONG_MAX but can be
   *        larger than INT_MAX
   *
   * @tparam T
   * @tparam std::enable_if<is_numeric<T>::value, bool>::type
   * @param buf
   * @param count
   * @param src
   * @param tag
   * @param comm
   * @param status
   */
  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Recv(T *buf, unsigned long count, int src = MPI_ANY_SOURCE,
                   int tag = MPI_ANY_TAG, Comm comm = worldComm,
                   Status *status = MPI_STATUS_IGNORE) {
    while (count > INT_MAX) {
      MPI_Recv(buf, INT_MAX, MPIX_TYPE<T>::type, src, tag, comm, status);
      count -= INT_MAX;
      buf += INT_MAX;
    }
    MPI_Recv(buf, (int)count, MPIX_TYPE<T>::type, src, tag, comm, status);
  }

  /**
   * @brief Receive a general data type following MPI standard
   *
   * @param buf
   * @param count
   * @param datatype
   * @param src
   * @param tag
   * @param comm
   * @param status
   */
  static void Recv(void *buf, int count, Datatype datatype,
                   int src = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG,
                   Comm comm = worldComm, Status *status = MPI_STATUS_IGNORE) {
    MPI_Recv(buf, count, datatype, src, tag, comm, status);
  }

  /**
   * @brief Receive a std::string
   *
   * @param str
   * @param src
   * @param tag
   * @param comm
   * @param status
   */

  static void Recv(std::string &str, int src = MPI_ANY_SOURCE,
                   int tag = MPI_ANY_TAG, Comm comm = worldComm,
                   Status *status = MPI_STATUS_IGNORE) {
    unsigned long len = 0;
    MPIX::Recv(len, src, tag, comm);
    if (len != 0) {
      str.resize(len);
      auto *buffer = &*str.begin();
      MPIX::Recv<>(buffer, len, src, tag, comm, status);
    } else {
      str.clear();
    }
  }

  /**
   * @brief Receive a std::vector containing fundamental data type
   *
   * @tparam T
   * @tparam A
   * @tparam std::enable_if<std::is_fundamental<T>::value, bool>::type
   * @param vec
   * @param src
   * @param tag
   * @param comm
   * @param status
   */
  template <
      typename T, typename A,
      typename std::enable_if<std::is_fundamental<T>::value, bool>::type = true>
  static void Recv(std::vector<T, A> &vec, int src = MPI_ANY_SOURCE,
                   int tag = MPI_ANY_TAG, Comm comm = worldComm,
                   Status *status = MPI_STATUS_IGNORE) {
    unsigned long len = 0;
    MPIX::Recv(len, src, tag, comm, status);
    if (len != 0) {
      vec.resize(len);
      MPIX::Recv(vec.data(), len, src, tag, comm, status);
    } else {
      vec.clear();
    }
  }

  static void Get_count(Status *status, Datatype datatype, int *count);

  static void Isend(void *buf, int count, Datatype datatype, Request *request,
                    int dest, int tag = MPI_DEFAULT_SEND_TAG,
                    Comm comm = worldComm);
  static void Isend(void *buf, int count, Datatype datatype, int dest, int tag,
                    Comm comm, Request *request);

  static void Irecv(void *buf, int count, Datatype datatype, Request *request,
                    int src = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG,
                    Comm comm = worldComm);
  static void Irecv(void *buf, int count, Datatype datatype, int src, int tag,
                    Comm comm, Request *request);

  static void Ibarrier(Comm comm, Request *request);
  static void Ibarrier(Request *request);

  static void Wait(Request *request, Status *status = MPI_STATUS_IGNORE);

  static void Waitall(int nrequests, Request *request,
                      Status *status = MPI_STATUSES_IGNORE);

  static void Waitany(int nrequests, Request *request, int *index,
                      Status *status = MPI_STATUSES_IGNORE);
  static void Waitany(int nrequests, Request *request, int &index,
                      Status *status = MPI_STATUSES_IGNORE);

  static void Probe(int source, int tag, Comm comm = worldComm,
                    Status *status = MPI_STATUS_IGNORE);

  static void Bcast(void *buf, int count, Datatype datatype,
                    int root = MPI_MASTER_PROCESS, Comm comm = worldComm);

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Bcast(T &value, int root = MPI_MASTER_PROCESS,
                    Comm comm = worldComm) {
    MPI_Bcast(&value, 1, MPIX_TYPE<T>::type, root, comm);
  }

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Bcast(T *buf, unsigned long count, int root = MPI_MASTER_PROCESS,
                    Comm comm = worldComm) {
    while (count > INT_MAX) {
      MPI_Bcast(buf, INT_MAX, MPIX_TYPE<T>::type, root, comm);
      count -= INT_MAX;
      buf += INT_MAX;
    }
    MPI_Bcast(buf, count, MPIX_TYPE<T>::type, root, comm);
  }

  template <typename T, typename A>
  static void Bcast(std::vector<T, A> &vec, int root = MPI_MASTER_PROCESS,
                    Comm comm = worldComm) {
    unsigned long len = vec.size();
    MPIX::Bcast(len, root, comm);
    if (len != 0) {
      vec.resize(len);
      MPIX::Bcast(vec.data(), len, root, comm);
    } else {
      vec.clear();
    }
  }

  static void Bcast(std::string &str, int root = MPI_MASTER_PROCESS,
                    Comm comm = worldComm);

  static void Bsend(void *buf, int count, Datatype datatype, int dest,
                    int tag = MPI_DEFAULT_SEND_TAG, Comm comm = worldComm);

  static void Buffer_attach(void *buffer, int size);

  static void Buffer_detach(void *buffer, int *size);

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Reduce(T &value, Op op, int root = MPI_MASTER_PROCESS,
                     Comm comm = worldComm) {
    MPI_Reduce(MPI_IN_PLACE, &value, 1, MPIX_TYPE<T>::type, op, root, comm);
  }

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Reduce(T *buf, unsigned long count, Op op,
                     int root = MPI_MASTER_PROCESS, Comm comm = worldComm) {
    while (count > INT_MAX) {
      MPI_Reduce(MPI_IN_PLACE, buf, INT_MAX, MPIX_TYPE<T>::type, op, root,
                 comm);
      count -= INT_MAX;
      buf += INT_MAX;
    }
    MPI_Reduce(MPI_IN_PLACE, buf, (int)count, MPIX_TYPE<T>::type, op, root,
               comm);
  }

  static void Reduce(void *sendbuf, void *recvbuf, int count, Datatype datatype,
                     Op op, int root = MPI_MASTER_PROCESS,
                     Comm comm = worldComm);

  static void Reduce(void *sendbuf, int count, Datatype datatype, Op op,
                     int root = MPI_MASTER_PROCESS, Comm comm = worldComm);

  template <typename T, typename A>
  static void Reduce(std::vector<T, A> &sendbuf, std::vector<T, A> &recvbuf,
                     Op op, int root = MPI_MASTER_PROCESS,
                     Comm comm = worldComm) {
    unsigned int len = sendbuf.size();
    // we don't check consistency of sendbuf size here
    // it should be user's responsibility to provide
    // the same size

    if (Rank == root) recvbuf.resize(len);
    if (&sendbuf == &recvbuf) {
      MPI_Reduce(MPI_IN_PLACE, sendbuf.data(), len, MPIX_TYPE<T>::type, op,
                 root, comm);
    } else {
      MPI_Reduce(sendbuf.data(), recvbuf.data(), len, MPIX_TYPE<T>::type, op,
                 root, comm);
    }
  }

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Allreduce(T &value, Op op, Comm comm = worldComm) {
    MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPIX_TYPE<T>::type, op, comm);
  }

  template <typename T,
            typename std::enable_if<is_numeric<T>::value, bool>::type = true>
  static void Allreduce(T *buf, unsigned long count, Op op,
                        Comm comm = worldComm) {
    while (count > INT_MAX) {
      MPI_Allreduce(MPI_IN_PLACE, buf, INT_MAX, MPIX_TYPE<T>::type, op, comm);
      count -= INT_MAX;
      buf += INT_MAX;
    }
    MPI_Allreduce(MPI_IN_PLACE, buf, (int)count, MPIX_TYPE<T>::type, op, comm);
  }

  static void Allreduce(void *sendbuf, void *recvbuf, int count,
                        Datatype datatype, Op op, Comm comm = worldComm);

  static void Allreduce(void *sendbuf, int count, Datatype datatype, Op op,
                        Comm comm = worldComm);

  template <typename T, typename A>
  static void Allreduce(std::vector<T, A> &sendbuf, std::vector<T, A> &recvbuf,
                        Op op, int root = MPI_MASTER_PROCESS,
                        Comm comm = worldComm) {
    unsigned int len = sendbuf.size();
    // we don't check consistency of sendbuf size here
    // it should be user's responsibility to provide
    // the same size
    recvbuf.resize(len);
    if (&sendbuf == &recvbuf) {
      MPI_Allreduce(MPI_IN_PLACE, sendbuf.data(), len, MPIX_TYPE<T>::type, op,
                    comm);
    } else {
      MPI_Allreduce(sendbuf.data(), recvbuf.data(), len, MPIX_TYPE<T>::type, op,
                    comm);
    }
  }

  static void Gather(void *sendbuf, int sendcnt, Datatype sendtype,
                     void *recvbuf, int recvcnt, Datatype recvtype,
                     int root = MPI_MASTER_PROCESS, Comm comm = worldComm);

  static void Scatter(void *sendbuf, int sendcnt, Datatype sendtype,
                      void *recvbuf, int recvcnt, Datatype recvtype,
                      int root = MPI_MASTER_PROCESS, Comm comm = worldComm);

  static void Allgather(void *sendbuf, int sendcnt, Datatype sendtype,
                        void *recvbuf, int recvcnt, Datatype recvtype,
                        Comm comm = worldComm);

  static void Allgatherv(void *sendbuf, int sendcount, Datatype sendtype,
                         void *recvbuf, int *recvcounts, int *displs,
                         Datatype recvtype, Comm comm = worldComm);

  static void Alltoall(void *sendbuf, int sendcount, Datatype sendtype,
                       void *recvbuf, int recvcount, Datatype recvtype,
                       Comm comm = worldComm);

  static void Alltoallv(void *sendbuf, int *sendcounts, int *sdispls,
                        Datatype sendtype, void *recvbuf, int *recvcounts,
                        int *recvdispls, Datatype recvtype,
                        Comm comm = worldComm);

  static void Sendrecv(void *sendbuf, int sendcnt, Datatype sendtype, int dest,
                       int sendtag, void *recvbuf, int recvcnt,
                       Datatype recvtype, int source, int recvtag,
                       Comm comm = worldComm,
                       Status *status = MPI_STATUS_IGNORE);

  static void Reduce_scatter(void *sendbuf, void *recvbuf, int *recvcounts,
                             Datatype datatype, Op op, Comm comm = worldComm);

  static double Wtime(void);

  static int split_job(int num_job);

  static std::vector<int> split_job(const std::vector<int> &num_job);

  static void Consecutive_exec(std::function<void()> f,
                               Comm comm = MPI_COMM_WORLD);

  static std::string read_ascii_file(const char file_path[]);
};

#endif
