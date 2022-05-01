FROM nvidia/cuda:11.1-runtime-ubuntu20.04
RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    
RUN apt update
RUN apt install python3 python3-pip -y

COPY esm_model_alphabet.pt .
COPY esm_model_args.pt .
COPY esm_model_state_dict.pt .

RUN pip3 install torch==1.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install fair-esm==0.4.0
RUN pip3 install tqdm
RUN pip3 install biopython
RUN pip3 install pandas

COPY generate_embeddings.py generate_embeddings.py
COPY sample.fasta sample.fasta
