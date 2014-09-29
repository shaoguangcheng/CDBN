#ifndef GLOBAL_H
#define GLOBAL_H

#ifndef DEBUGMSG
#define DEBUGMSG(msg) cout << "line: " << __LINE__ \
    /*<< ", function: " << __func__ */<< \
    ", file: " << __FILE__ \
    << ", message: " << msg << endl
#endif

#include <stdexcept>

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
            throw std::logic_error("logic error occur while using handle class");
    }

    const T*& operator -> () const{
        if(base)
            return base;
        else
            throw std::logic_error("logic error occur while using handle class");
    }

    T& operator * () {
        if(base)
            return *base;
        else
            throw std::logic_error("logic error occur while using handle class");
    }

    const T& operator * () const{
        if(base)
            return *base;
        else
            throw std::logic_error("logic error occur while using handle class");
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

#endif // GLOBAL_H
