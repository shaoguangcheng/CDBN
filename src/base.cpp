#include <iostream>
#include <stdlib.h>

#include <libconfig.h++>

#include "base.h"

using namespace libconfig;

////////////////////////////// net //////////////////////////////
net::net()
{}

net::net(const string& configFile)
    : configFile(configFile)
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
}

/////////////////////////////// option /////////////////////////////
option::option()
{}

option::option(const string& configFile)
    :configFile(configFile)
{
    parse();
}

void option::parse()
{}
