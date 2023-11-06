# starter code for a2

Add the corresponding (one) line under the ``[to fill]`` in ``def forward()`` of the class for ffnn.py and rnn.py

Feel free to modify other part of code, they are just for your reference.

---

One example on running the code:`

**FFNN**

``python ffnn.py --hidden_dim 10 --epochs 1 ``
``--train_data ./training.json --val_data ./validation.json``

```python ffnn.py --hidden_dim 128 --epochs 5 --train_data training.json --val_data validation.json```

```python ffnn.py --hidden_dim 128 --epochs 5 --train_data training.json --val_data validation.json --test_data test.json --do_test```


**RNN**

``python rnn.py --hidden_dim 32 --epochs 10 ``
``--train_data training.json --val_data validation.json``

