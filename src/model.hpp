//note : here must not include model.h

///////////////////// CRBMModel /////////////////////////
template <class T, int DIM>
void CRBMModel<T, DIM>::writeToFile(const string &fileName) const
{
    ofstream out(fileName.c_str());
    if(out.bad()){
        DEBUGMSG("Unable to write model to file" + fileName);
        exit(EXIT_FAILURE);
    }

    out << W << endl;
    out << biasV << endl;
    out << biasH << endl;
    out << top << endl;

    out.close();
}

template <class T, int DIM>
void CRBMModel<T, DIM>::loadFromFile(const string &filename)
{
    ifstream in(filename.c_str());
    if(in.bad()){
        DEBUGMSG("Unable to read model from file" + filename);
        exit(EXIT_FAILURE);
    }

    in >> W;
    in >> biasV;
    in >> biasH;
    in >> top;

    in.close();
}

template <class T, int DIM>
ostream& operator << (ostream& out, const CRBMModel<T, DIM>& crbm)
{
    out << crbm.W << endl;
    out << crbm.biasV << endl;
    out << crbm.biasH << endl;
    out << crbm.top << endl;

    return out;
}

template <class T, int DIM>
istream& operator >> (istream& in, CRBMModel<T, DIM>& crbm)
{
    in >> crbm.W;
    in >> crbm.biasV;
    in >> crbm.biasH;
    in >> crbm.top;

    return in;
}

///////////////////// CDBNModel //////////////////////////
template <class T, int DIM>
void CDBNModel<T, DIM>::writeToFile(const string& fileName) const
{
    ofstream out(fileName.c_str());

    if(out.bad()){
        DEBUGMSG("Unable to read model from file" + fileName);
        exit(EXIT_FAILURE);
    }

    out << model.size() << endl;

    size_t size = model.size();
    for(int i=0; i<size; ++i){
        out << model[i];
    }

    out.close();
}

template <class T, int DIM>
void CDBNModel<T, DIM>::loadFromFile(const string& fileName)
{
    model.clear();

    ifstream in(fileName.c_str());
    if(in.bad()){
        DEBUGMSG("Unable to read model from file" + fileName);
        exit(EXIT_FAILURE);
    }

    size_t size;
    in >>size;

    CRBMModel<T, DIM> crbm;
    int i = 0;
    while(i < size){
        in >> crbm;
        model.push_back(crbm);
        ++i;
    }

    in.close();
}
