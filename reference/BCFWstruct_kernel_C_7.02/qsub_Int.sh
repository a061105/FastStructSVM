#!/bin/bash
#
# sample lsf bsub to run an interactive job, optionally on a selected host.
#
# pick a host to land on.
host=${1:-tulgb004}

#
# the -Is says you want an interactive session
# the s says you want a terminal session.
#
# shared_int is the "shared interactive queue"
if [ -z $LSB_BATCH_JID ]; then
  set -x
  bsub \
  -Is \
  -n 1 \
  -q shared_int \
  -m $host \
  -W 3500 \
  /bin/bash
fi 
