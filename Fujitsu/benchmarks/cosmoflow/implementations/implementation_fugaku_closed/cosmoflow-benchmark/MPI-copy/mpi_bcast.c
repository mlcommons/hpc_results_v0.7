#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <limits.h>
#include <sys/stat.h>

long int getFileSize(const char* fileName)
{
    struct stat statBuf;

    if (stat(fileName, &statBuf) == 0)
        return statBuf.st_size;

    return -1L;
}

int main(int argc, char **argv){

  char *data;
  int filesize, tmpi;
  char *param;
  char *filepath;
  FILE *fp;
  int roots_flag, groupID;
  int global_rank, global_procs;
  MPI_Comm bcast_comm;
  int bcast_rank, bcast_procs, bcast_root;
  int ierr;

  ierr = MPI_Init(&argc, &argv);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "MPI init error\n");
    return -1;
  }

  ierr = MPI_Comm_size(MPI_COMM_WORLD, &global_procs);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "comm_size(world) error\n");
    return -1;
  }

  ierr = MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "comm_rank(world) error\n");
    return -1;
  }

  // Get environment variables
  param = getenv("Roots");
  if(param != NULL){
    roots_flag = atoi(param) > 0 ? 1 : 0;
  }else{
    fprintf(stderr, "Environment variable Roots is needed\n");
    return -1;
  }
  param = getenv("GroupID");
  if(param != NULL){
    groupID = atoi(param);
  }else{
    fprintf(stderr, "Environment variable GroupID is needed\n");
    return -1;
  }
  filepath = getenv("TargetFilePath");
  if(filepath == NULL){
    fprintf(stderr, "Environment variable TargetFilePath is needed\n");
    return -1;
  }

  // Create sub groups
  ierr = MPI_Comm_split(MPI_COMM_WORLD, groupID, groupID, &bcast_comm);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "comm_split error\n");
    return -1;
  }

  ierr = MPI_Comm_size(bcast_comm, &bcast_procs);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "comm_size(bcast_comm) error\n");
    return -1;
  }

  ierr = MPI_Comm_rank(bcast_comm, &bcast_rank);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "comm_rank(bcast_comm) error\n");
    return -1;
  }

  // Get root rank and filesize
  if(roots_flag > 0){
    filesize = getFileSize(filepath);
  }

  tmpi = (roots_flag > 0) ? bcast_rank : 0;
  ierr = MPI_Allreduce(&tmpi, &bcast_root, 1, MPI_INT, MPI_SUM, bcast_comm);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "Allreduce error\n");
    return -1;
  }
  ierr = MPI_Bcast(&filesize, 1, MPI_INT, bcast_root, bcast_comm);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "data size: Bcast error\n");
    return -1;
  }

  // Malloc
  data = (char *)malloc(filesize * sizeof(char));
  if(data == NULL){
    fprintf(stderr, "Malloc error\n");
    return -1;
  }

  // Read file
  if(roots_flag > 0){
    fp = fopen(filepath, "rb");
    if(fp == NULL){
      fprintf(stderr, "%s cannot be opened\n", filepath);
      return -1;
    }
    fread(data, sizeof(char), filesize, fp);
    fclose(fp);  
  }

  // Data Bcast
  ierr = MPI_Bcast(data, filesize, MPI_CHAR, bcast_root, bcast_comm);
  if( ierr != MPI_SUCCESS ){
    fprintf(stderr, "data: Bcast error\n");
    return -1;
  }

  if(roots_flag == 0){
    fp = fopen(filepath, "wb");
    if(fp == NULL){
      fprintf(stderr, "%s cannot be opened\n", filepath);
      return -1;
    }
    fwrite(data, sizeof(char), filesize, fp);
    fclose(fp);  
  }

  // Finalize
  free(data);
  MPI_Comm_free(&bcast_comm);
  MPI_Finalize();

  return 0;
}
