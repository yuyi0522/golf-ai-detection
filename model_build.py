from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Masking
from keras.metrics import MeanSquaredError, RootMeanSquaredError, MeanAbsoluteError, MeanAbsolutePercentageError


def model_build(max_timestep, feature):
    # 生成模型
    # Adding the first input LSTM layer and Dropout regularisation
    model = Sequential()
    model.add(Masking(mask_value=0.00))
    model.add(LSTM(64, input_shape=(max_timestep, feature), return_sequences=True, dtype='float32'))
    # Adding a second LSTM layer and Dropout regularisation
    model.add(LSTM(32, input_shape=(max_timestep, 64), return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and Dropout regularisation
    model.add(LSTM(16, input_shape=(max_timestep, 32), return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and Dropout regularisation
    model.add(LSTM(8, input_shape=(max_timestep, 16), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=150, activation='relu'))
    model.add(Dense(units=feature*300, activation="linear"))
    # Compiling
    model.compile(optimizer='adam', loss='mse',
                  metrics=[MeanSquaredError(), RootMeanSquaredError(), MeanAbsoluteError(), MeanAbsolutePercentageError()])
    model.build((1, max_timestep, feature))
    model.summary()

    return model

