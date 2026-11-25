# -*- coding: utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings('ignore')

from model import get_model

model = get_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()
