# from pytorch images
FROM pytorch/pytorch

# add this directory to docker image's root directory
COPY ./ /tianchi

# set workplace
WORKDIR /tianchi

# run some comand
RUN pip install -r requirement.txt -i https://pypi.tuna.tsinghua.edu.cn/simple some-package