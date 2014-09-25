#ifndef BASE_H
#define BASE_H

/**
 * @brief In this file, I try to define some basic structure to represent Convolution Deep Belief Network, including
 *        input layer, convolution layer, pooling layer and the whole net.
 */

#include <string>
#include <vector>
#include <cstddef>
#include <stdexcept>

using namespace std;

#ifndef DEBUGMSG
#define DEBUGMSG(msg) cout << "line: " << __LINE__ \
    /*<< ", function: " << __func__ */<< \
    ", file: " << __FILE__ \
    << ", message: " << msg << endl
#endif

/**
 * @brief The layerType enum define the input layer type
 */
enum layerType{
    GAUSSIAN = 0,
    BERNOULLI
};

/**
 * @brief The netType enum define the type of whole network
 */
enum netType{
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
 * @brief The inputLayer class  derived input layer class
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
    convLayer(const string& name, int nFeatureMap, int kernelSize) :
        layer(name), nFeatureMap(nFeatureMap), kernelSize(kernelSize){}

    convLayer(const convLayer& l) : layer(l), nFeatureMap(l.nFeatureMap), kernelSize(l.kernelSize){}
    convLayer& operator = (const convLayer& l){
        layer::operator =(l);
        nFeatureMap = l.nFeatureMap;
        kernelSize = l.kernelSize;

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
};

/**
 * @brief The poolingLayer class define the structure of pooling layer
 */
class poolingLayer : public layer
{
public :
    poolingLayer(){}
    poolingLayer(const string& name, int scale) : layer(name), scale(scale){}
    poolingLayer(const poolingLayer& l) : layer(l), scale(l.scale){}
    poolingLayer& operator = (const poolingLayer& l){
        layer::operator =(l);
        scale = l.scale;

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
};


/**
 * define the template class to manage the base class pointer
 */
template <class T>
class handle{
public :
    handle() : base(NULL), use(new size_t(1)){}
    handle(T* base) : base(base), use(new size_t(1)){}
    handle(const handle& h) : base(h.base), use(h.use){
        ++*use;
    }
    handle& operator = (const handle& h){
        ++*h.use;
        decreaseUse();
        base = h.base;
        use  = h.use;

        return * this;
    }

    ~handle(){
        decreaseUse();
    }

    T*& operator -> (){
        if(base)
            return base;
        else
            throw logic_error("");
    }

    const T*& operator -> () const{
        if(base)
            return base;
        else
            throw logic_error("");
    }

    T& operator * () {
        if(base)
            return *base;
        else
            throw logic_error("");
    }

    const T& operator * () const{
        if(base)
            return *base;
        else
            throw logic_error("");
    }

private :
    void decreaseUse(){
        if(1 == *use){
            delete base;
            delete use;
            return;
        }

        --*use;
    }

private :
    T* base;
    size_t* use;
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

    /**
     * @brief type the type of network which can be UNSUPERVISED or SUPERVISED
     */
    netType type;

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
