#define ILOUSESTL
using namespace std;

#include <ilcplex/ilocplex.h>
#include <vector>
#include "matrix.h"
#include "mex.h"

ILOSTLBEGIN
typedef vector<IloNumArray>    NumMatrix;
typedef vector<IloNumVarArray> NumVarMatrix;
typedef vector<IloBoolVarArray> BoolVarMatrix;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
#define param       prhs[0]
#define data        prhs[1]
#define optimal     plhs[0]
    
    if (nrhs != 2) {
        mexErrMsgTxt("Two input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgTxt("Too many output arguments.");
    }
    
    mxArray *s_in       = mxGetField(data, 0, "s");
    mxArray *x_in       = mxGetField(data, 0, "x");    
    size_t n            = mxGetM(x_in);
    size_t N            = mxGetN(x_in);
    size_t m            = mxGetM(s_in);
    
    mxArray *epsilon_in = mxGetField(param, 0, "epsilon");
    mxArray *pnorm_in   = mxGetField(param, 0, "pnorm");
    mxArray *C_in       = mxGetField(param, 0, "C");
    mxArray *d_in       = mxGetField(param, 0, "d");
    mxArray *h_in       = mxGetField(param, 0, "h");
    mxArray *H_in       = mxGetField(param, 0, "H");
    mxArray *W_in       = mxGetField(param, 0, "W");
	mxArray *alpha_in	= mxGetField(param, 0, "alpha");
    mxArray *delta_in   = mxGetField(param, 0, "delta");
    mxArray *set_in     = mxGetField(param, 0, "set_theta");
    
    double pnorm        = mxGetScalar(pnorm_in);
    size_t nd           = mxGetM(d_in);
    size_t nh           = mxGetM(h_in);
    size_t ne           = mxGetNumberOfElements(epsilon_in);
    
    /* Initialization */
    const char *field_names[] = {"theta","objective","diagnosis"};
    optimal = mxCreateStructMatrix(ne, 1, 3, field_names); 
    
    IloEnv env;
    try {
        /* Define Additional Fixed Vectors */
        NumMatrix x(N);
        NumMatrix s(N);
        IloNumArray epsilon(env,ne);
        NumMatrix C(nd);
        IloNumArray d(env,nd);
        IloNumArray h(env,nh);
        NumMatrix H(nh);
        NumMatrix W(nh);
        NumMatrix Ctr(m);
        NumMatrix Htr(m);
        NumMatrix Wtr(n);
        IloNum alpha;
        IloNum delta;
        for (int i=0; i < N; i++) {
            x[i]        = IloNumArray(env,n);
            s[i]        = IloNumArray(env,m);
            for (int j=0; j<n; j++){
                x[i][j] = mxGetPr(x_in)[j+i*n];
            }
            for (int j=0; j<m; j++){
                s[i][j] = mxGetPr(s_in)[j+i*m];
            }
        }
        for (int i=0; i < ne; i++){
            epsilon[i]  = mxGetPr(epsilon_in)[i];
            if (epsilon[i] == 0)  // It is numerically more stable and efficient to replace 0 with small value   
                epsilon[i] = 1e-10;
        }
        for (int i=0; i < nd; i++) {
            C[i]        = IloNumArray(env,m);
            d[i]        = mxGetPr(d_in)[i];
            for (int j=0; j<m; j++){
                C[i][j] = mxGetPr(C_in)[j*nd+i];
            }
        }
        for (int i=0; i < nh; i++) {
            H[i]        = IloNumArray(env,m);
            W[i]        = IloNumArray(env,n);
            h[i]        = mxGetPr(h_in)[i];
            for (int j=0; j<m; j++){
                H[i][j] = mxGetPr(H_in)[j*nh+i];
            }
            for (int j=0; j<n; j++){
                W[i][j] = mxGetPr(W_in)[j*nh+i];
            }
        }
        for (int i=0; i<m; i++){
            Ctr[i]      = IloNumArray(env,nd);
            Htr[i]      = IloNumArray(env,nh);
            for (int j=0; j<nd; j++){
                Ctr[i][j] = mxGetPr(C_in)[i*nd+j];
            }
            for (int j=0; j<nh; j++){
                Htr[i][j] = mxGetPr(H_in)[i*nh+j];
            }
        }
        for (int i=0; i<n; i++){
            Wtr[i]      = IloNumArray(env,nh);
            for (int j=0; j<nh; j++){
                Wtr[i][j] = mxGetPr(W_in)[i*nh+j];
            }
        }
        alpha = mxGetScalar(alpha_in);
        if (delta_in != NULL) 
            delta = mxGetScalar(delta_in);
        else 
            delta = 0;
        if (alpha == 0)
            alpha = 1 / N;
        IloNum bigM = 2;

        for (int ep = 0; ep < ne; ep++) {
            IloModel model(env);
            
            /* Define Decision Variables */
            IloNumVar       lambda(env,-IloInfinity,IloInfinity);          
            IloNumVar       tau(env,-IloInfinity,IloInfinity);
            IloNumVarArray  r(env,N,-IloInfinity,IloInfinity);
            IloNumVarArray  theta(env,n,-IloInfinity,IloInfinity);
            NumVarMatrix    zx_1(N);
            NumVarMatrix    zx_2(N);
            NumVarMatrix    zs_1(N);
            NumVarMatrix    zs_2(N);
            NumVarMatrix    gamma(N);
            NumVarMatrix    mu_1(N);
            NumVarMatrix    mu_2(N);
            NumVarMatrix    phi_1(N);
            NumVarMatrix    phi_2(N);
            for (int i=0; i < N; i++) {
                zx_1[i]     = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
                zx_2[i]     = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
                zs_1[i]     = IloNumVarArray(env,m,-IloInfinity,IloInfinity);
                zs_2[i]     = IloNumVarArray(env,m,-IloInfinity,IloInfinity);
                gamma[i]    = IloNumVarArray(env,nh,0,IloInfinity);
                mu_1[i]     = IloNumVarArray(env,nh,0,IloInfinity);
                mu_2[i]     = IloNumVarArray(env,nh,0,IloInfinity);
                phi_1[i]    = IloNumVarArray(env,nd,0,IloInfinity);
                phi_2[i]    = IloNumVarArray(env,nd,0,IloInfinity);
            } 
            
            /* Declare Constraints */
            for (int i=0; i<N; i++) {
                model.add(IloScalProd(zs_1[i],s[i]) - IloScalProd(phi_1[i],d) + IloScalProd(zx_1[i],x[i]) - IloScalProd(mu_1[i],h) - IloScalProd(gamma[i],h) <= r[i] + tau + delta);
                model.add(IloScalProd(zs_2[i],s[i]) - IloScalProd(phi_2[i],d) + IloScalProd(zx_2[i],x[i]) - IloScalProd(mu_2[i],h) <= r[i]);
                for (int j=0; j<m; j++){
                    model.add(zs_1[i][j] == IloScalProd(Ctr[j],phi_1[i]) - IloScalProd(Htr[j],mu_1[i]) - IloScalProd(Htr[j],gamma[i]) );
                    model.add(zs_2[i][j] == IloScalProd(Ctr[j],phi_2[i]) - IloScalProd(Htr[j],mu_2[i]) );
                }
                for (int j=0; j<n; j++){
                    model.add(zx_1[i][j] == IloScalProd(Wtr[j],mu_1[i]) + IloScalProd(Wtr[j],gamma[i]) );
                    model.add(zx_2[i][j] == IloScalProd(Wtr[j],mu_2[i]));
                    model.add(theta[j]   == IloScalProd(Wtr[j],gamma[i]));
                }
            }
            if ( pnorm == 1 ) {
                NumVarMatrix    s1(N);
                NumVarMatrix    s2(N);
                NumVarMatrix    s3(N);
                NumVarMatrix    s4(N);
                for (int i=0; i<N; i++){
                    s1[i]       = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
                    s2[i]       = IloNumVarArray(env,m,-IloInfinity,IloInfinity);
                    s3[i]       = IloNumVarArray(env,n,-IloInfinity,IloInfinity);
                    s4[i]       = IloNumVarArray(env,m,-IloInfinity,IloInfinity);
                    for (int j=0; j<n; j++) {
                        model.add( zx_1[i][j] <= s1[i][j] );
                        model.add(-zx_1[i][j] <= s1[i][j] );
                        model.add( zx_2[i][j] <= s3[i][j] );
                        model.add(-zx_2[i][j] <= s3[i][j] );
                    }
                    for (int j=0; j<m; j++){
                        model.add( zs_1[i][j] <= s2[i][j] );
                        model.add(-zs_1[i][j] <= s2[i][j] );
                        model.add( zs_2[i][j] <= s4[i][j] );
                        model.add(-zs_2[i][j] <= s4[i][j] );
                    }
                    model.add( IloSum(s1[i]) + IloSum(s2[i]) <= lambda );
                    model.add( IloSum(s3[i]) + IloSum(s4[i]) <= lambda );
                }
            }
            if ( pnorm == 2 ) { 
                IloNumVarArray  s1(env,N,0,IloInfinity);
                IloNumVarArray  s2(env,N,0,IloInfinity);
                for (int i=0; i<N; i++){
                    model.add( IloScalProd(zx_1[i],zx_1[i]) + IloScalProd(zs_1[i],zs_1[i]) <= IloSquare(s1[i]) );
                    model.add( IloScalProd(zx_2[i],zx_2[i]) + IloScalProd(zs_2[i],zs_2[i]) <= IloSquare(s2[i]) );
                    model.add( s1[i] <= lambda );
                    model.add( s2[i] <= lambda );
                }
            }
            if ( mxIsInf(pnorm) ){
                for (int i=0; i<N; i++){
                    for (int j=0; j<n; j++) {
                        model.add( zx_1[i][j] <= lambda );
                        model.add(-zx_2[i][j] <= lambda );
                        model.add( zx_2[i][j] <= lambda );
                        model.add(-zx_2[i][j] <= lambda );
                    }
                    for (int j=0; j<m; j++){
                        model.add( zs_1[i][j] <= lambda );
                        model.add(-zs_1[i][j] <= lambda );
                        model.add( zs_2[i][j] <= lambda );
                        model.add(-zs_2[i][j] <= lambda );
                    }
                }
            } 

            /* Constrains corresponds to the set Theta */
            if (set_in != NULL) {                 
                mxArray *center_in  = mxGetField(set_in, 0, "center");
                mxArray *radius_in  = mxGetField(set_in, 0, "radius");
                mxArray *norm_in    = mxGetField(set_in, 0, "pnorm");
                mxArray *A_in       = mxGetField(set_in, 0, "A");
                mxArray *b_in       = mxGetField(set_in, 0, "b");
                if (center_in != NULL && radius_in != NULL && norm_in != NULL) {
                    IloNumArray center(env,n);
                    for (int i=0; i < n; i++)
                        center[i]       = mxGetPr(center_in)[i];
                    IloNum radius       = mxGetScalar(radius_in);
                    double nrm          = mxGetScalar(norm_in);
                    if ( nrm == 1 ) {
                        IloNumVarArray  rr(env,n,0,IloInfinity);
                        for (int i=0; i<n; i++){
                            model.add( theta[i] - center[i] <= rr[i] );
                            model.add( center[i] - theta[i] <= rr[i] );
                        }
                        model.add( IloSum(rr) <= radius );
                    }
                    if ( nrm == 2 ) {
                        IloNumVarArray  rr(env,n,0,IloInfinity);
                        for (int i=0; i<n; i++){
                            model.add( theta[i] - center[i] == rr[i] );
                        }
                        model.add( IloScalProd(rr,rr) <= radius * radius );
                    }
                    if ( mxIsInf(nrm) ) {
                        for (int i=0; i<n; i++){
                            model.add( theta[i] - center[i] <= radius );
                            model.add( center[i] - theta[i] <= radius );
                        }
                    }
                }
                if (A_in != NULL && b_in != NULL) {
                    size_t nb       = mxGetM(b_in);
                    NumMatrix A(nb);
                    for (int i=0; i < nb; i++) {
                        A[i]        = IloNumArray(env,n);
                        for (int j=0; j<n; j++)
                            A[i][j] = mxGetPr(A_in)[j+i*n];
                    }
                    IloNumArray b(env,nb);
                    for (int i=0; i < nb; i++)
                        b[i]        = mxGetPr(b_in)[i];
                    // for (int i=0; i < nb; i++)
                    //     model.add( IloScalProd(A[i],theta) >= b[i] ); 
                    model.add( IloSum(theta) == -1 );
                }
            }
            else{  /* Implementation of ||theta||_infty = 1 */
                IloBoolVarArray b1(env,n);
                IloBoolVarArray b2(env,n);
                IloNum bigM = 2;
                for (int i=0; i<n; i++){
                    model.add(theta[i] >= bigM * (b1[i]-1) + 1);
                    model.add(theta[i] <= bigM * (1-b2[i]) - 1);
                }
                model.add(IloSum(b1) + IloSum(b2) == 1); 
            }

            /* Declare Objective */
            IloExpr objective(env);
            objective = tau + 1 /alpha * (lambda * epsilon[ep] + IloSum(r) / N);
            model.add(IloMinimize(env,objective));
            objective.end();
            
            /* Solve the Problem */
            IloCplex cplex(model);
            cplex.setOut(env.getNullStream());
            cplex.solve();

            /* Save the Results */
            mxArray *theta_out       = mxCreateDoubleMatrix(n, 1, mxREAL); 
            mxArray *objective_out   = mxCreateDoubleMatrix(1, 1, mxREAL); 
            mxArray *diagnosis_out   = mxCreateDoubleMatrix(1, 1, mxREAL); 
            double* out_theta        = mxGetPr(theta_out);
            double* out_objective    = mxGetPr(objective_out);
            double* out_diagnosis    = mxGetPr(diagnosis_out);
            *out_diagnosis           = cplex.getStatus();
            mxSetField(optimal, ep, "diagnosis", diagnosis_out);
            if (*out_diagnosis != 2){
                    continue;
            }
            *out_objective           = cplex.getObjValue();
            for (int i=0; i < n; i++)
            	out_theta[i]         = cplex.getValue(theta[i]);
            mxSetField(optimal, ep, "theta", theta_out);
            mxSetField(optimal, ep, "objective", objective_out);
            
            /* Close CPLEX and Model & Free Memory! */
            cplex.end();
            model.end();
        }
    }
    catch (IloException& ex) {
        mexErrMsgTxt ("Error Stoc");
    }
    catch (...) {
        mexErrMsgTxt ("Error");
    }
    env.end();
    return;
}