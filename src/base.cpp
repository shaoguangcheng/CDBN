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

    // read the net structure
    try{
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
                convLayer conv;

                conv.name = nameTmp;
                singleLayer.lookupValue("nFeatureMap", conv.nFeatureMap);
                singleLayer.lookupValue("kernelSize", conv.kernelSize);

                layers.push_back(handle<layer>(conv.clone()));

                continue;
            }

            if("pooling" == nameTmp){ //parse pooling layer
                poolingLayer pooling;

                pooling.name = nameTmp;
                singleLayer.lookupValue("scale", pooling.scale);

                layers.push_back(handle<layer>(pooling.clone()));

                continue;
            }

            // if reach here, there must be some errors in configuration file
            DEBUGMSG("undefined layer type");
            exit(EXIT_FAILURE);
        }
    }
    catch(const SettingNotFoundException& notFound){
        DEBUGMSG("Setting not Found");
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
{
    Config cfg;

    try{
        cfg.readFile(configFile.c_str());
    }
    catch(const FileIOException& IOError){
        DEBUGMSG("IO error occur when reading the configuration file");
        exit(EXIT_FAILURE);
    }
    catch(const ParseException& parseError){
        cerr << "Parse error at " << parseError.getFile()
             << ", line " << parseError.getLine()
             << "Error : " << parseError.getError() << endl;
        exit(EXIT_FAILURE);
    }

    try{
        cfg.lookupValue("dataname", dataName);
        cfg.lookupValue("outputPath", outputPath);
        cfg.lookupValue("biasMode", biasMode);
        NDIM      = cfg.lookup("NDIM");
        sparsity  = cfg.lookup("sparsity");
        lambda1   = cfg.lookup("lambda1");
        lambda2   = cfg.lookup("lambda2");
        alpha     = cfg.lookup("alpha");
        nEpoch    = cfg.lookup("nEpoch");
        batchSize = cfg.lookup("batchSize");
        nCD       = cfg.lookup("nCD");
    }
    catch(const SettingNotFoundException& notFound){
        DEBUGMSG("Setting not Found");
        exit(EXIT_FAILURE);
    }

}

//////////////////////////////////////////////////////////////////////
// overload the output stream
ostream& operator << (ostream& out, const inputLayer& l)
{
    out << "name : " << l.name << ", type : ";
    if(GAUSSIAN == l.type)
        out << "gaussian";
    if(BERNOULLI == l.type)
        out << "bernoulli";

    return out;
}

ostream& operator << (ostream& out, const convLayer& l)
{
    out << "name : " << l.name
        << ", nFeatureMap : " << l.nFeatureMap
        << ", kernelSize  : " << l.kernelSize;

    return out;
}

ostream& operator << (ostream& out, const poolingLayer& l)
{
    out << "name : " << l.name
        << ", scale : " << l.scale;

    return out;
}

ostream& operator << (ostream& out, const net& n)
{
    string nameTmp;
    net::netStructure ns = n.layers;

    out << "================= network structure =============" << endl;
    out << "=========== " << n.size() << " layer " << "=======" << endl;
    out << "network type : ";
    if(UNSUPERVISED == n.type)
        out << "unsupervised" << endl;
    if(SUPERVISED == n.type)
        out << "supervised" << endl;

    for(size_t i=0; i<n.size(); ++i){
        nameTmp = ns[i]->name;

        if("input" == nameTmp){
            inputLayer* inputPtr = dynamic_cast<inputLayer*>(ns[i]->clone());
            out << *inputPtr << endl;
            delete  inputPtr;

            continue;
        }

        if("convolution" == nameTmp){
            convLayer* convPtr = dynamic_cast<convLayer*>(ns[i]->clone());
            out << *convPtr << endl;
            delete convPtr;

            continue;
        }

        if("pooling" == nameTmp){
            poolingLayer* poolingPtr = dynamic_cast<poolingLayer*>(ns[i]->clone());
            out << *poolingPtr << endl;
            delete poolingPtr;

            continue;
        }
    }

    out << "==================================================" << endl;

    return out;
}
ostream& operator << (ostream& out, const option& n)
{
    out << "========== network options ==========" << endl;
    out << "dataname   : " << n.dataName << endl;
    out << "outputPath : " << n.outputPath << endl;
    out << "NDIM       : " << n.NDIM << endl;
    out << "biasMode   : " << n.biasMode << endl;
    out << "sparsity   : " << n.sparsity << endl;
    out << "lambda1    : " << n.lambda1 << endl;
    out << "lambda2    : " << n.lambda2 << endl;
    out << "alpha      : " << n.alpha << endl;
    out << "nEpoch     : " << n.nEpoch << endl;
    out << "batchSize  : " << n.batchSize << endl;
    out << "nCD        : " << n.nCD << endl;
    out << "======================================" << endl;

    return out;
}
