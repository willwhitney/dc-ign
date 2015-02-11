ID = '1';
load(['RESULTS/preds_' num2str(ID) '.mat']);
preds = x;
load(['RESULTS/targets_' num2str(ID) '.mat']);
targets = x;

bsize = size(preds,4);

for i=1:bsize
   inf_img = preds(:,:,:,i);
   gt_img = targets(:,:,:,i);
   figure(1);clf; imshow(gt_img); title('Ground Truth');
   figure(2);clf; imshow(inf_img); title('Inferred');
   pause
end
