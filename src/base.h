#ifndef BASE_H
#define BASE_H

/**
 * @brief In this file, I try to define some basic structure to represent Convolution Deep Belief Network, including
 *        input layer, convolution layer, pooling layer and the whole net.
 */

#include <string>
#include <vector>
#include <cstddef>

#include "global.h"

using namespace std;

/**
 * @brief The layerType enum define the input layer type
 */
enum layerType{
    GAUSSIAN = 0,
    BERNOULLI
};

/**
 * @brief The poolingType enum define the pooling type
 */
enum poolingType{
    MAX = 0,
    MEAN,
    STOCHASTIC
};

/**
 * @brief The netType enum define the type of whole network
 */
enum netType{ // unused
    UNSUPERVISED = 0,
    SUPERVISED
};

/**
 * @brief The layer class  the base class for all kinds of layers
 */
class layer
{
public :
    layer(){}
    layer(const string& name) : name(name){}
    layer(const layer& l) : name(l.name){}
    layer& operator = (const layer& l){
        name = l.name;
        return *this;
    }

    virtual ~layer(){}

    virtual layer* clone() const{
        return new layer(*this);
    }

public :
    string name;
};

/**
 * @brief The inputLayer class  derived from input layer class
 */
class inputLayer : public layer
{
public :
    inputLayer() : layer(){}
    inputLayer(const string& name, layerType type) : layer(name), type(type){}
    inputLayer(const inputLayer& l) : layer(l), type(l.type){}
    inputLayer& operator = (const inputLayer& l){
        layer::operator =(l);
        type = l.type;

        return *this;
    }

    ~inputLayer(){}

    inputLayer* clone() const{
        return new inputLayer(*this);
    }

public :
    /**
     * @brief type the type of input layer, maybe GAUSSIAN or BERNOULLI
     */
    layerType type;
};

/**
 * @brief The convLayer class deine the structure of convolution layer
 */
class convLayer : public layer
{
public :
    convLayer() : layer(){}
    convLayer(const string& name, int nFeatureMap, int kernelSize, int stride)
        : layer(name), nFeatureMap(nFeatureMap), kernelSize(kernelSize), stride(stride){}

    convLayer(const convLayer& l)
        : layer(l), nFeatureMap(l.nFeatureMap), kernelSize(l.kernelSize), stride(l.stride){}
    convLayer& operator = (const convLayer& l){
        layer::operator =(l);
        nFeatureMap = l.nFeatureMap;
        kernelSize = l.kernelSize;
        stride = l.stride;

        return *this;
    }

    ~convLayer(){}

    convLayer* clone() const{
        return new convLayer(*this);
    }


public :
    /**
     * @brief nFeatureMap the feature map number for convolution layer
     */
    int nFeatureMap;

    /**
     * @brief kernelSize the kernel size for convolution operation
     */
    int kernelSize;

    /**
     * @brief stride sliding length of each step for convolution kernel
     */
    int stride;
};

/**
 * @brief The poolingLayer class define the structure of pooling layer
 */
class poolingLayer : public layer
{
public :
    poolingLayer(){}
    poolingLayer(const string& name, int scale, poolingType type)
        : layer(name), scale(scale), type(type){}
    poolingLayer(const poolingLayer& l)
        : layer(l), scale(l.scale), type(l.type){}
    poolingLayer& operator = (const poolingLayer& l){
        layer::operator =(l);
        scale = l.scale;
        type  = l.type;

        return *this;
    }

    ~poolingLayer(){}

    poolingLayer* clone() const{
        return new poolingLayer(*this);
    }

public :
    /**
     * @brief scale scale size from convolution layer to pooling layer
     */
    int scale;

    poolingType type;
};

/**
 * @brief The net class  the structure of network
 */
class net{
public :
    net();
    net(const string& configFile);

    inline size_t size() const {return layers.size();}

public :
    typedef vector<handle<layer> > netStructure;

    /**
     * @brief layers save the structure of network
     */
    netStructure layers;

private :
    /**
     * @brief parse parse the configuration file to obtain the structure of network
     */
    void parse(const string& configFile);
};

/**
 * @brief The option class the miscellanous option for the network
 */
class option{
public :
    option();
    option(const string& configFile);

private :
    /**
     * @brief parse parse the configuration file to obtain the structure of network
     */
    void parse(const string &configFile);

public :
    string dataName;
    string outputPath;
    string biasMode;

    int NDIM;
    int nEpoch;
    int batchSize;
    int nCD;

    double sparsity;
    double lambda1;
    double lambda2;
    double alpha;
};

ostream& operator << (ostream& out, const inputLayer& l);
ostream& operator << (ostream& out, const convLayer& l);
ostream& operator << (ostream& out, const poolingLayer& l);
ostream& operator << (ostream& out, const net& n);
ostream& operator << (ostream& out, const option& n);

#endif // BASE_H
