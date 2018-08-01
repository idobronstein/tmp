
for entreg in 1 #1 0.1 10 0.01
do
  python train_cifar100.py --entropy_reg ${entreg} >& train10K_ent_manyKiter${entreg}.log 
done

   
    
 
 
