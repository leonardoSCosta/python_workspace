import sys
from PySide2.QtCore import QPoint, Qt
from PySide2.QtGui import QPainter
from PySide2.QtWidgets import QMainWindow, QApplication
from PySide2.QtCharts import QtCharts
from PySide2 import QtCore, QtGui, QtWidgets


class TestChart(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        self.lineSeries = QtCharts.QLineSeries()
        self.lineSeries.setName("Grafico de RainFlow")

        self.chart = QtCharts.QChart()
        self.chart.addSeries(self.lineSeries)
        self.chart.setTitle("Algoritmo de RainFlow")

        self.maxX = 25
        self.maxY = 20
        self.xTickValue = 1
        self.yTickValue = 5

        self.axisX = QtCharts.QValueAxis()
        self.axisX.setRange(0,self.maxX)
        self.axisX.setTickCount(self.maxX / self.xTickValue + 1)
        self.chart.setAxisX(self.axisX, self.lineSeries)

        self.axisY = QtCharts.QValueAxis()
        self.axisY.setRange(-self.maxY, self.maxY)
        self.axisY.setTickCount(2 * self.maxY / self.yTickValue + 1)
        self.chart.setAxisY(self.axisY, self.lineSeries)



        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignBottom)

        self.chartView = QtCharts.QChartView(self.chart)
        self.chartView.setRenderHint(QPainter.Antialiasing)

        self.setCentralWidget(self.chartView)

    def adjustYValue(self, yValue):
        print(yValue)
        yValue = self.yTickValue * round(float(yValue) / float(self.yTickValue))
        print(yValue)
        return yValue

        digits = []

        while yValue > 0:
            digits.append(yValue % 10)
            yValue /= 10

        if digits[len(digits) - (len(digits)-1)] > 5:
            
            yValue = (digits[-1] + 1) * pow(10, (len(digits)-1))
        else:
            yValue = (digits[-1]) * pow(10, (len(digits)-1))

        return yValue


    def mousePressEvent(self, event):
        if event.type() == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == QtCore.Qt.MouseButton.LeftButton:

                self.scaleX = self.maxX/self.chart.plotArea().width()
                self.scaleY = 2*self.maxY/self.chart.plotArea().height()

                p = event.pos() - self.chart.plotArea().topLeft().toPoint()# relative to widget
                p.setX(round(p.x()*self.scaleX))
                yAdjust = -round(p.y()*self.scaleY - self.maxY)

                yAdjust = self.adjustYValue(int(yAdjust))
                
                p.setY(yAdjust)

                self.lineSeries.append(p)
            elif event.button() == QtCore.Qt.MouseButton.RightButton:
                self.lineSeries.remove(self.lineSeries.count()-1)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = TestChart()
    window.show()
    window.resize(440, 300)

    sys.exit(app.exec_())