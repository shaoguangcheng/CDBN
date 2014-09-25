#ifndef GLOBAL_H
#define GLOBAL_H

#ifndef DEBUGMSG
#define DEBUGMSG(msg) cout << "line: " << __LINE__ \
    /*<< ", function: " << __func__ */<< \
    ", file: " << __FILE__ \
    << ", message: " << msg << endl
#endif

#endif // GLOBAL_H
