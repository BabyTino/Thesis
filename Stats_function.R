z.prop = function(nd,nc,na,ne){
        
 numerator = (nd/na) - (nc/ne)
 p.common = (nd+nc) / (na+ne)
 denominator = sqrt(p.common * (1-p.common) * (1/na + 1/ne))
 z.prop.ris = numerator / denominator
 p_value=2*(1-pnorm(abs(z.prop.ris)))
 return(list(z.prop.ris=z.prop.ris, p_value=p_value))
}

z.prop()

