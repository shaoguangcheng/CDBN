TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ../src/main.cpp \
    ../src/base.cpp \
    ../src/test.cpp \
    ../src/util.cpp \
    ../src/model.hpp

HEADERS += \
    ../src/base.h \
    ../src/model.h \
    ../src/test.h \
    ../src/util.h \
    ../src/global.h

INCLUDEPATH += /opt/Blitz++/include
LIBS += -L/opt/Blitz++/lib/ -lblitz

INCLUDEPATH += /opt/libconfig/include
LIBS += -L/opt/libconfig/lib -lconfig++

OBJECTS_DIR = ./tmp
