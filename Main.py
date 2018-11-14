import matplotlib.pyplot as plt
import datahandler as dh
import ANN as ann
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

    # Training
    model = ann.train(x_train, y_train, 3, 200)
    print('MSE data latih : {}'.format(ann.evaluate(x_train, y_train, model)))
    ann.save_model(model, 'model')

    # Testing
    file_test = 'DataTest.xlsx'
    data_test = load_excel(file_test)

    model = ann.load_model('model')
    model.compile(loss='mean_squared_error', optimizer='adam')

    data_test = dh.data_norm(data_test)
    x_test, y_test = dh.generate_series(data_test, 3)
    print('MSE data uji : {}'.format(ann.evaluate(x_test, y_test, model)))
    result = ann.predict(x_test, model)

    plot_timeseries(result)
