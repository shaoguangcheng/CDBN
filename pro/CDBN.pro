TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += ../src/main.cpp \
    ../src/base.cpp \
    ../src/model.cpp

HEADERS += \
    ../src/base.h \
    ../src/model.h

INCLUDEPATH += /opt/Blitz++/include
LIBS += -L/opt/Blitz++/lib -lblitz

OBJECTS_DIR = ./tmp
