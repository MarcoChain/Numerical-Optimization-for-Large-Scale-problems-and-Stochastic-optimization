function [xk, lambdak, sk, fxk, k ] = interior_point_method(c, epsilon, kmax, n)

e = ones(n,1);
r = e';
k = 0;

x_hat = r'*1./(r*r');
lambda_hat = 1./(r*r')*r*c;
s_hat = c - r'*lambda_hat;

deltino_x = max(-1.5*min(x_hat), 0);
deltino_s = max(-1.5*min(s_hat), 0);

x_hat = x_hat + deltino_x*e;
s_hat = s_hat + deltino_s*e;


deltino_hat_x = 0.5*(x_hat'*s_hat)/(e'*s_hat);
deltino_hat_s = 0.5*(x_hat'*s_hat)/(e'*x_hat);

xk = x_hat + deltino_hat_x*e;
lambdak = lambda_hat;
sk = s_hat + deltino_hat_s*e;

muk = (xk' * sk) / n;
mu0 = muk;
I = speye(n);

F = spalloc(2*n+1, 2*n+1, 5*n);
F(1, 1:n)=r;
F(2:n+1, n+1) = r';
F(2:n+1, n+2:2*n+1) = I;

while k<kmax
    %2.a
    Xk = spdiags(xk, 0, n, n);
    Sk = spdiags(sk, 0, n, n);
     
    F(n+2:2*n+1, 1:n) = Sk;
    F(n+2:2*n+1, n+2:2*n+1) = Xk;
    %spy(F)
    r1 = 1-r*xk;
    r2 = c - r'*lambdak - sk;
    r3 = - Xk*Sk*e;
    
    BB_aff = [r1; r2; r3];
    XX_aff = F\BB_aff;
    %L = ichol(F, struct('type','nofill','michol','on'));
    %[XX_aff, flag, err, iter, res] = pcg(F, BB_aff, 1e-3, 10000, L, L');
    delta_xk_aff = XX_aff(1:n);
    delta_sk_aff = XX_aff(n+2:2*n+1);
    
    %2.b
    z1 = -xk./delta_xk_aff;
    alpha_aff_p = min(1, min(z1(delta_xk_aff<0)));
    if isempty(alpha_aff_p)
        alpha_aff_p = 1;
    end
    
    z2 = -sk./delta_sk_aff;
    alpha_aff_d = min(1, min(z2(delta_sk_aff<0)));
     if isempty(alpha_aff_d)
        alpha_aff_d = 1;
     end
    
    %2.c
    muk_aff= (xk + alpha_aff_p*delta_xk_aff)'*(sk+alpha_aff_d*delta_sk_aff)/n;
    sigmak = (muk_aff/muk)^3;

    %2.d
    delta_Xk_aff = spdiags(delta_xk_aff, 0, n, n);
    delta_Sk_aff = spdiags(delta_sk_aff, 0, n, n);
    BB = [r1; ...
          r2; ...
          r3 - delta_Xk_aff*delta_Sk_aff*e + sigmak*muk*e];
    XX = F\BB;
    
    %2.e
    delta_xk= XX(1:n);
    delta_lambdak= XX(n+1);
    delta_sk= XX(n+2:2*n+1);
    
    z1 = -xk./delta_xk;
    alphakmax_p =  min(z1(delta_xk<0));
    z2 = -sk./delta_sk;
    alphakmax_d = min(z2(delta_sk<0));
    etak = max(1-muk, 0.9);
    alphak_p = min(1, etak*alphakmax_p);
    if isempty(alphak_p)
        alphak_p = 1;
    end
    alphak_d = min(1, etak*alphakmax_d);
    if isempty(alphak_d)
        alphak_d = 1;
    end
    
    %2.f
    xk = xk + alphak_p*delta_xk;
    lambdak = lambdak + alphak_d*delta_lambdak;
    sk = sk + alphak_d*delta_sk;
    
    %2.g
    k = k+1;
    muk = xk'*sk / n;
    if  muk<mu0*epsilon
        fxk = c'*xk;
        return
    end
    k
end

fxk = c'*xk;
end




