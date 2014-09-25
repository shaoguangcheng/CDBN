#include <iostream>
#include <stdlib.h>

#include <libconfig.h++>

#include "base.h"

using namespace libconfig;

////////////////////////////// net //////////////////////////////
net::net()
{}

net::net(const string& configFile)
{
    parse(configFile);
}

void net::parse(const string &configFile)
{
    Config cfg;

    try{
        cfg.readFile(configFile.c_str());
    }
    catch(const FileIOException& IOError){
        cerr << "I/O Error occur while reading the configuration file0" << endl;
        exit(EXIT_FAILURE);
    }
    catch(const ParseException& parseError){
        cerr << "Parse Error at " << parseError.getFile() << ", line " << parseError.getLine()
             << ", Error : " << parseError.getError() << endl;
        exit(EXIT_FAILURE);
    }

    // read network type
    string typeTmp = cfg.lookup("net.type");
    if("supervised" == typeTmp){
        type = SUPERVISED;
    }
    else{
        if("unsupervised" == typeTmp){
            type = UNSUPERVISED;
        }
        else{
            DEBUGMSG("undefined network type");
            exit(EXIT_FAILURE);
        }
    }

    const Setting& root = cfg.getRoot();
    const Setting& allLayer = root["net"]["layers"];
    int count = allLayer.getLength();

    string nameTmp;
    for(int i=0; i<count; ++i){
        const Setting& singleLayer = allLayer[i];
        singleLayer.lookupValue("name", nameTmp);

        if("input" == nameTmp){ //parse input layer
            string typeTmp;
            inputLayer input;

            input.name = nameTmp;

            singleLayer.lookupValue("type", typeTmp);
            if("gaussian" == typeTmp){
                input.type = GAUSSIAN;
            }
            else{
                if("bernoulli" == typeTmp){
                    input.type = BERNOULLI;
                }
                else{
                    DEBUGMSG("undefined input type");
                    exit(EXIT_FAILURE);
                }
            }

            layers.push_back(handle<layer>(input.clone())); // save the input layer

            continue;
        }

        if("convolution" == nameTmp){ //parse convolution layer
            continue;
        }

        if("pooling" == nameTmp){ //parse pooling layer
            continue;
        }

        // if reach here, there must be some errors in configuration file
        DEBUGMSG("undefined layer type");
        exit(EXIT_FAILURE);
    }

}

/////////////////////////////// option /////////////////////////////
option::option()
{}

option::option(const string& configFile)
{
    parse(configFile);
}

void option::parse(const string& configFile)
{}

//////////////////////////////////////////////////////////////////////
ostream& operator << (ostream& out, const net& n)
{
    return out;
}
ostream& operator << (ostream& out, const option& n)
{
    return out;
}
