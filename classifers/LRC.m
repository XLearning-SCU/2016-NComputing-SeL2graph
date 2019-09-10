function [Rec] = LRC(tr_dat, tt_dat, trls, ttls)
[Aq,Q]=qr(tr_dat,0);
Q=pinv(Q);
C = Q*Aq';
coef=C*tt_dat;
clear Aq Q C ;
%inference via sparse coding classifier with local information embbeding
for indTest = 1:size(tt_dat,2)
    ID(indTest) = IDcheck(tr_dat, coef(:,indTest), tt_dat(:,indTest), trls);
end
cornum      =   sum(ID==ttls);
% recognition rate
Rec = [cornum/length(ttls)];
fprintf(['The classification result of LRC is about ' num2str(Rec) '\n']);