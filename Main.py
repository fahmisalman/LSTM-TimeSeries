import matplotlib.pyplot as plt
import datahandler as dh
# import ANN as ann
import LSTM as lstm
import xlrd


def plot_timeseries(x):
    plt.plot(x)
    plt.show()


def load_excel(filename):
    wb = xlrd.open_workbook(filename)
    sh = wb.sheet_by_index(0)
    data = [sh.cell(i, 0).value for i in range(sh.nrows)]
    return data


if __name__ == '__main__':

    file_train = 'DataTrain.xlsx'
    data_train = load_excel(file_train)

    # Normalisasi
    data_train = dh.data_norm(data_train)
    x_train, y_train = dh.generate_series(data_train, 3)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Training
    model = lstm.train(x_train, y_train, epoch=100)
    print('MSE data latih : {}'.format(lstm.evaluate(x_train, y_train, model)))
    lstm.save_model(model, 'model')

    # Testing
    file_test = 'DataTest.xlsx'
    data_test = load_excel(file_test)

    model = lstm.load_model('model')
    model.compile(loss='mean_squared_error', optimizer='adam')

    data_test = dh.data_norm(data_test)
    x_test, y_test = dh.generate_series(data_test, 3)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    print('MSE data uji : {}'.format(lstm.evaluate(x_test, y_test, model)))
    result = lstm.predict(x_test, model)

    plot_timeseries(result)
