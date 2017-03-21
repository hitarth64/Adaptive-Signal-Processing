// x_n: Observation/ Measurement
// d_n: Actual data 
// mu: error modification parameter
// Function implements an IIR recursive/ adaptive filter
function[a_n,b_n,err] =iirecursive(p,q,mu,x_n,d_n)
    // Adaptive IIR filter
    // p,q : degree of numerator and denominator of IIR filter
    
    a_n(:,1) = zeros(p,1); //Initializing coefficients to zeros
    b_n(:,1) = zeros(q+1,1); //Initializing coefficients to zeros
    y_n = zeros(length(x_n),1); //Predicted data
    yf_n = zeros(length(x_n),1); //Derivatives initialized to zeros
    xf_n = zeros(length(x_n),1); //Derivatives initialized to zeros
    
    for i = 1:length(x_n)-1
        yvec_n_1 = zeros(p,1); // Needed for calculation of y[n]
        xvec_n = zeros(q+1,1); // Needed for calculation of y[n]
        xf_vec_n = yvec_n_1; // Needed when we are trying to evaluate xf[n]
        yf_vec_n = xf_vec_n; // Needed when we are trying to evaluate yf[n]
        
        xvec_n(1) = x_n(i);
        
        for k=1:p
            if i-k>0 then 
                yvec_n_1(k) = y_n(i-k);
                yf_vec_n(k) = yf_n(i-k);
                xf_vec_n(k) = xf_n(i-k);
            end
        end
        
        for k=1:q
            if i-k>0 then xvec_n(k+1) = x_n(i-k);
            end
        end

        y_n(i) = yvec_n_1' * a_n(:,i) + xvec_n' * b_n(:,i);
        err(i) = d_n(i) - y_n(i);
        yf_n(i) = y_n(i) + yf_vec_n' * a_n(:,i);
        xf_n(i) = x_n(i) + xf_vec_n' * a_n(:,i);
        a_n(:,i+1) = a_n(:,i) + mu*err(i)*yf_vec_n;

        xf_vec = zeros(q+1,1); // Needed for final step where we need xf from n-1 to n-q-1.
        
        for k=1:q+1
            if i-k>0 then xf_vec(k) = xf_n(i-k+1);
            end
        end
        
        b_n(:,i+1) = b_n(:,i) + mu*err(i)*xf_vec; 
    end
    
endfunction
