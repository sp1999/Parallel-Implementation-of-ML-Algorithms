#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <omp.h>
#include <math.h>

// Equation: y = 2x+10
const int N = 100000;
int max_iter = 100000;
double error;
double alpha0 = 0.01; // learning rate
double alpha1 = 0.01; // learning rate
double epsilon = 0.00001; // convergence 

void update_theta(double* x, double* y, double *t0, double *t1)
{
    double grad0 = 0, grad1 = 0;
    #pragma omp parallel for default(none) shared(grad0, grad1, x, y, t0, t1)
    for(int i = 0; i < N; i++)
    {
        // #pragma omp atomic update
        grad0 += (y[i] - (1/(1 + exp(-(*t0) - (*t1) *x[i]))))*(1/(1 + exp(-(*t0) - (*t1) *x[i])))*(1 - (1/(1 + exp(-(*t0) - (*t1) *x[i]))));
        grad1 += (y[i] - (1/(1 + exp(-(*t0) - (*t1) *x[i]))))*(1/(1 + exp(-(*t0) - (*t1) *x[i])))*(1 - (1/(1 + exp(-(*t0) - (*t1) *x[i]))))*x[i];
    }
    grad0 /= N;
    grad1 /= N;

    // Updating theta
    double temp0 = (*t0) + alpha0*grad0;
    double temp1 = (*t1) + alpha1*grad1;
    *t0 = temp0;
    *t1 = temp1;
}

double total_error(double* x, double* y, double t0, double t1)
{
    double total = 0;
    #pragma omp parallel for default(none) shared(total, x, y, t0, t1)
    for(int i = 0; i < N; i++)
    {
        #pragma omp atomic update
        total += ((1/(1 + exp(-t0 - t1*x[i])) - y[i]) * (1/(1 + exp(-t0 - t1*x[i])) - y[i]));
    }
    return total;
}

int main(int argc, char* argv[])
{
    srand(time(0));
    int num_threads = 2; // Number of threads
    omp_set_num_threads(num_threads);
    // x data and y_data (training set)
    double x[N], y[N];

    // initialise these vectors
    // y = 2x+10
    #pragma omp parallel for default(none) shared(x,y)
    for(int i = 0; i < N; i++)
    {
        x[i] = (double)rand()/RAND_MAX;
        y[i] = 2*x[i] +10;
        // Adding random noise to the data
        y[i] += (double)rand()/RAND_MAX;
    }

    // convergence criteria
    double t0,t1; //theta0 and theta1

    //gradient_descent(alpha,ep,x,y,100,theta0,theta1); // max_iter = 100
    //------Start of Batch Gradient Descent---------------//
    bool converged = false;
    int iter = 0;      // max_iter = 100;

    // initialising the theta
    t0 = 0;
    t1 = 0;

    // total error, theta
    double J = total_error(x, y, t0, t1);

    // iterate loop 
    while(!converged)
    {
        // Compute gradient and update theta
        update_theta(x,y,&t0,&t1);
        error=total_error(x,y,t0,t1);
        if (error < J)
        {
            alpha0 *= 1.1;
            alpha1 *= 1.1;
        }
        else
        {
            alpha0 *= 0.5;
            alpha1 *= 0.5;
        }
        if(abs(J-error) <= epsilon)
        {
            printf("Converged, iterations = %d\n", iter+1);
            converged=true;
        }
        J = error;
        iter += 1;
        if(iter == max_iter)
        {
            printf("Max iterations crossed\n");
            converged = true;
        }
    }
    printf("%lf %lf\n", t0, t1);
    return 0;
}