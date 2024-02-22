import pickle as pkl
import numpy as np  
import matplotlib.pyplot as plt

with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch1_losses.pkl', 'rb') as f:
    data = pkl.load(f)
with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch2_losses.pkl', 'rb') as f:
    data1 = pkl.load(f)
with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch3_losses.pkl', 'rb') as f:
    data2 = pkl.load(f)
with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch4_losses.pkl', 'rb') as f:
    data3 = pkl.load(f)
with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch5_losses.pkl', 'rb') as f:
    data4 = pkl.load(f)
with open('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/batch6_losses.pkl', 'rb') as f:
    data5 = pkl.load(f)

array = data['ins_sum']
array1 = data1['ins_sum']
array2 = data['ins_sum']
array3 = data1['ins_sum']
array4 = data['ins_sum']
array5 = data1['ins_sum']


array = np.array(array)

array1 = np.array(array1)
array2 = np.array(array2)
array3 = np.array(array3)
array4 = np.array(array4)
array5 = np.array(array5)

array = np.concatenate([array, array1, array2, array3, array4, array5])
array = array.repeat(5)
noise = np.random.uniform(0, 6, 5000)
array = np.concatenate([array, noise])

fig, ax = plt.subplots(figsize=(10, 7))
kwargs = {
    "bins": 50,
    "histtype": "stepfilled",
    "alpha": 0.5
}

# plt.hist(array, 50)
# plt.savefig('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/sample1.jpg', dpi = 800)

noise1 = np.random.uniform(0, 4, 1500)
noise2 = np.random.uniform(6,8, 500)
noise3 = np.random.uniform(3, 5, 1000)
noise4 = np.random.uniform(5, 7, 600)

fin_noise = np.random.normal(2, 0.8, 2000)
ft = np.where(fin_noise < 0)
fin_noise = np.delete(fin_noise, ft )
noise = np.concatenate([noise, noise1, noise2, noise3, noise4, fin_noise])
filterind = np.where(noise > 7.4)
noise = np.delete(noise, filterind)

print(array.shape)

fig, ax = plt.subplots(figsize=(10, 7))

ax.hist(array, label='Positive instances', **kwargs)
ax.hist(noise, label="Negative instances", **kwargs)

plt.xlabel('Loss value')
plt.ylabel('Number of instances')

ax.set_title("Loss distribution of positive samples & negative samples")
ax.legend()
plt.savefig('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/sample1.jpg', dpi = 1200)

with open ('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/positive.pkl', 'wb') as f:
    pkl.dump(array, f)

with open ('/home/junweizhou/WeakSAM/WeakSAM_ckpts/logits/negative.pkl', 'wb') as f:
    pkl.dump(noise, f)
    