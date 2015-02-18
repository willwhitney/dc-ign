TOTALFACES = 0;%5230;
num_train_batches = 200;%5000;
num_test_batches =  0;%TOTALFACES-num_train_batches;

for i=1:num_train_batches
   basedir = '../facemachine/CNN_DATASET_FINET/';
%    load([basedir 'face_' num2str(i) '/data.mat']);
%    img = double(img)/255.0;
%    save(['DATASET/training/' num2str(i) '.mat'], 'img');
  allimg = dir([basedir 'face_' num2str(i) '/*.png']);
  cnt=1;
 mkdir(['DATASET/FT_training/face_' num2str(i)]); 
  for f=allimg'
     fname = f.name;
     img = imread([basedir 'face_' num2str(i) '/' fname]);
     imwrite(img,['DATASET/FT_training/face_' num2str(i) '/' num2str(cnt) '.png']);
     cnt = cnt + 1;
  end
    i
end

for i=1:num_test_batches
   basedir = '../facemachine/CNN_DATASET/';
%    load([basedir 'face_' num2str(i) '/data.mat']);
%    img = double(img)/255.0;
%    save(['DATASET/training/' num2str(i) '.mat'], 'img');
  allimg = dir([basedir 'face_' num2str(i) '/*.png']);
  cnt=1;
 mkdir(['DATASET/FT_test/face_' num2str(i)]); 
  for f=allimg'
     fname = f.name;
     img = imread([basedir 'face_' num2str(i) '/' fname]);
     imwrite(img,['DATASET/FT_test/face_' num2str(i) '/' num2str(cnt) '.png']);
     cnt = cnt + 1;
  end
    i
end