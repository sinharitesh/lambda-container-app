FROM public.ecr.aws/lambda/python:3.6

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install torch==1.7.0+cpu torchvision==0.8.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install fastai==2.3.0 
RUN pip install pillow==8.2
   
COPY app.py export.pkl  ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]

