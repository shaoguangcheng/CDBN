TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ../src/main.cpp \
    ../src/base.cpp \
    ../src/test.cpp \
    ../src/util.cpp \
    ../src/model.hpp \
    ../src/CRBM.hpp \
    ../src/matrixOperation.cpp

HEADERS += \
    ../src/base.h \
    ../src/model.h \
    ../src/test.h \
    ../src/util.h \
    ../src/global.h \
    ../src/CRBM.h \
    ../src/matrixOperation.h

INCLUDEPATH += /opt/Blitz++/include
LIBS += -L/opt/Blitz++/lib/ -lblitz

INCLUDEPATH += /opt/libconfig/include
LIBS += -L/opt/libconfig/lib -lconfig++

OBJECTS_DIR = ./tmp
