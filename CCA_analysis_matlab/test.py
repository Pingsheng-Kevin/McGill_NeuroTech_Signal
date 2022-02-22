import matlab.engine
import matlab
import numpy as np
from scipy.io import loadmat
from scipy.signal import filtfilt, cheby1


engine = matlab.engine.start_matlab()
data = loadmat('Sub02.mat')['data']
data = data.astype(float)
# print(data.shape)
freq_list = [8.0 + i*0.25 for i in range(32)]
freq_list.remove(12)

data_test = data[:, :, :, 3]
data_template = data[:, :, :, :2]
data_template = np.mean(data_template, axis=3)
results = []
predictions = []
onset = 100
for tar in range(len(freq_list)):
    signal = matlab.double(data_test[:, :, tar].T[onset+35:int(onset+35+250*4), :].tolist())
    beta, alpha = cheby1(N=2, rp=0.3, Wn=[7/125.0, 90/125.0], btype='band', output='ba')
    rho_array = []
    for i in range(len(freq_list)):
        template = filtfilt(beta, alpha, data_template[:, :, i]).T[onset+35:int(onset+35+250*4), :]
        template = matlab.double(template.tolist())
        rho = engine.FBCCA_IT(signal, freq_list[i], 7.0, 88.0, 5.0, template, 1.0, 0.5, 5.0, 250.0, 2.0)
        rho_array.append(rho)
    print(np.argmax(rho_array))
    predictions.append(np.argmax(rho_array))
    results.append(int(np.argmax(rho_array) == tar))
print(np.sum(results)/len(results))
print(predictions)
# data_template = engine.mean(matlab.double(data_template), 4)



