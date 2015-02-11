TOTALFACES = 5230;
num_train_batches = 5000;
num_test_batches =  TOTALFACES-num_train_batches;

for i=1:num_train_batches
   basedir = '../facemachine/CNN_DATASET/';
%    load([basedir 'face_' num2str(i) '/data.mat']);
%    img = double(img)/255.0;
%    save(['DATASET/training/' num2str(i) '.mat'], 'img');
  allimg = dir([basedir 'face_' num2str(i) '/*.png']);
  cnt=1;
 mkdir(['DATASET/training/face_' num2str(i)]); 
  for f=allimg'
     fname = f.name;
     img = imread([basedir 'face_' num2str(i) '/' fname]);
     imwrite(img,['DATASET/training/face_' num2str(i) '/' num2str(cnt) '.png']);
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
 mkdir(['DATASET/test/face_' num2str(i)]); 
  for f=allimg'
     fname = f.name;
     img = imread([basedir 'face_' num2str(i) '/' fname]);
     imwrite(img,['DATASET/test/face_' num2str(i) '/' num2str(cnt) '.png']);
     cnt = cnt + 1;
  end
    i
end