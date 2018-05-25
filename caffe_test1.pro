QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle
INCLUDEPATH += /usr/local/cuda-8.0/include
INCLUDEPATH += /home/kobayashi/caffe/include
QMAKE_CXXFLAGS += -std=c++11

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

LIBS += /home/kobayashi/caffe/build/lib/libcaffe.so
LIBS += /usr/lib/x86_64-linux-gnu/libglog.so
LIBS += /usr/lib/x86_64-linux-gnu/libprotobuf.so
LIBS += /usr/lib/x86_64-linux-gnu/libboost_system.so
LIBS += /usr/local/cuda-8.0/lib64/libcudnn.so


# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp
