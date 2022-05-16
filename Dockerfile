FROM public.ecr.aws/lambda/python:3.9

#RUN /var/lang/bin/python3.9 -m pip install --upgrade pip

#ENV PYTHONNUSERSITE=1

COPY predict.py ./
#COPY requirements.txt ./
ADD arabert ./arabert

COPY FullModel.h5 /opt/ml/model/                                                                                                                                                                                            

#RUN cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 python3.9 -m pip install
#RUN  pip3 install --no-cache-dir -r requirements.txt -t /usr/lib/python3.9/dist-packages/ 
#RUN python3.9 -m pip install -r requirements.txt
#RUN cat requirements.txt | xargs -n 1 -L 1 pip install

#RUN mkdir -p /scripts
COPY script.sh ./
COPY requirements3.txt ./
RUN chmod +x script.sh
RUN ./script.sh

#ENV PYTHONPATH "${PYTHONPATH}:/project"
#ENV PYTHONUNBUFFERED 1

#RUN yum -y update 

#RUN yum -y install git

#RUN git clone https://github.com/aub-mind/arabert.git

CMD ["predict.lambda_handler"]