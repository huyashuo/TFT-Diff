## Generate Code

~~~bash
Generate the core running part in \Experiments\chb_multiple_classes.ipynb or demo.py
run command：python demo.py
~~~

## Generate Data or Train Data

~~~bash
After running the previous step, the generated data is obtained in \chb_exp, and the test data and train data are obtained in \chb_exp\sample
~~~

## Metric Code

~~~bash
Evaluate the core running in \Experiments\metric_pytarch.ipynb and Experiments\metric_sensorflow.ipynb

Data replacement for evaluation section:
Example:
Replace the data path in ori_data = np.load('../chb_exp/ori_data_1.npy') with \Data\aug_train\full-0.2-200-raw.npy
Replace \Data\aug_train\full-0.2-200.npy with the data path in fake_data = np.load('../chb_exp/ddpm_fake_1_eeg.npy')
~~~

## Classic Code

~~~bash
Classification Core Run \Experiments\aug.ipynb
Replace the data in the classification section as above
~~~
