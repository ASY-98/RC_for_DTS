%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Ce programme utilise la méthode du reservoir computing pour apprendre à reconstituer un %%
%%%%%%%% jeu de données expérimentales. Il est basé sur un code d'Artur Perevalov de %%%%%%%%
%%%%% l'université du Maryland. Modifié par Adrien Saurety (ENS Lyon) sous la direction%%%%%%
%%d'Henri-Claude Nataf (ISTerre, Université Grenoble Alpes). Nécessite la toolbox Satistical%
%%%%%%%%%%%% and Optimisation toolbox pour le programme chaostest de Ahmed Bensaia%%%%%%%%%%% 

%%%%%% Récupération des données et préparations des fichiers

addpath('/data/geodynamo/natafh/Dynamo/DTSOmega/Traitement/modes_Denys');
addpath('/data/geodynamo/natafh/Dynamo/DTSOmega/Traitement/Magnetic/Heterogeneities/Programs');
addpath('/data/geodynamo/natafh/Dynamo/DTSOmega/Traitement/Manips_DTSNUM');

filename='150708S12X39'; %fichier à traiter

[history,tdeb,tfin,f_o,f_i,t_num,data_minus_motif,ideb_tot,ifin_tot,ReadMe_Data] = read_corrected_data(0,filename);
moy=50;
i=1;

for k=3:6:51
	ch_D_n=['BD+' int2str(k) '_Q'];
	ch_D_s=['BD-' int2str(k) '_Q'];
	ch_G_n=['BG+' int2str(k) '_Q'];
	ch_G_s=['BG-' int2str(k) '_Q'];
	disp ch_G_s ;
	[iD_n]=get_channel_index(ch_D_n,history.channels.selected_saved);%les channel name et channel lists hémisphère nord
	[iG_n]=get_channel_index(ch_G_n,history.channels.selected_saved);
	[iD_s]=get_channel_index(ch_D_s,history.channels.selected_saved);%les channel name et channel lists hémisphère sud
	[iG_s]=get_channel_index(ch_G_s,history.channels.selected_saved);


	if (iD_n~=0) && (iD_s~=0) && (iG_n~=0) && (iG_s~=0) %on ne récupère que les enregistrements "complets"

	       	disp 'ok';
		[data_T_n,data_P_n]=perform_rotation('150708','S12','X39',ch_D_n,ch_G_n,data_minus_motif(:,iD_n),data_minus_motif(:,iG_n)); %récupération de B thêta et de B phi à partir de la composante droite et gauche 
		[data_T_s,data_P_s]=perform_rotation('150708','S12','X39',ch_D_s,ch_G_s,data_minus_motif(:,iD_s),data_minus_motif(:,iG_s));
		[data_T_sym,~]=symmetrize_data('BT',data_T_n,data_T_s); %on récupère la partie symétrique du champs pour les trois composantes
		[data_P_sym,~]=symmetrize_data('BP',data_P_n,data_P_s);	

		nb=size(data_P_sym,1); 	% nombre de points
		if i==1
			data=zeros((floor(nb/moy))+1,1);
		end
		data_P_m_sym=movmean(data_P_sym,moy) %on lisse et on échantillonne 
		data_T_m_sym=movmean(data_T_sym,moy);
		data_P_me_sym=data_P_m_sym(1:moy:nb);
		data_T_me_sym=data_T_m_sym(1:moy:nb);

		%on a pris que le champs en thêta ici mais on peut ajouter les autres composantes

		data_P_fnial=zeros(floor(nb/moy)+1,1); %on normalise
		mea=mean(data_P_me_sym);
		data_P_final=(data_P_me_sym(:)-mea)/((mean((data_P_me_sym(:)-mea).^2)).^(1/2));

		data(:,i)=data_P_final(:);
		nbe=size(data,1); %nombre de points échantillonner
		i=i+1;
	end
end


%%%%%% Paramètres de l'apprentissage

start=floor(nbe/30);			% une petite partie des données = phase de start
training_length=floor((2*nbe)/3);	% une grosse partie = phase d'entrainement
test_length=nbe-(start+training_length); % le reste phase de prédiction pure pour voir si ça a marché

Utr=data(1:1:start+training_length,:);
Ute=data(1:1:start+training_length+test_length,:);

N=3500;			 % reservoir size
reg=0.0001;        	 % regularisation parameter
seed=27;		 % choisir une seed "qui marche bien" (dépendance des résultats à la façon de générer aléatoirement la matrice)
trans = 20;              % transient time - time we remove from the beginning of the training part
rng(seed);               % seed random number generator if want to reproduce the same Adjacency
alpha=0.955;             % leakage parameter
p = 0.08;                % connection probability
scale_rho = 0.2;         % input strength
	             	
train = training_length+start;      % training time
dim = size(Utr,2);                  % number of coordinates
scale = scale_rho*ones(1,dim);      % input strength for each coordinate
coord=3                             % coordinate to plot

%rho = spectrale radius, on cherchait sa valeur optimale donc on teste plusieurs valeurs

for rho=[1.4,1.3,1.2,1]
		
	U = Utr;

%% Generating RC

	% Trying creating adjacency matrix until it has a normal largest eigenvalue (not NaN)
	while true
	    A = sparse(double(rand(N) < p));        % apply connection probability
	    A(A~=0) = 2*rand(1,length(find(A)))-1;  % randomize nonzero entries
	    max_eig = abs(eigs(A,1));
	    if ~isnan(max_eig)
	        A = rho*A/max_eig;                  % scale spectral radius
	        break
	    end
	end
	
	% generating input weight for all channels
	Win = (2*rand(N,dim)-1)*diag(scale);        % input weights
	
%% Training phase

	X = zeros(train+1, N);                      % preallocate initial RC state
	X(1,:) = 2*rand(1,N)-1;
	
	for n = 1:train    
	    X(n+1,:) = (1-alpha) * X(n,:) + alpha*tanh(X(n,:)*A' + U(n,:)*Win');   % reservoir with input
	
	end

	init = X(train+1,:);        % initial state for prediction
	X = X((trans+1):train,:);   % throw away transient
	U = U((trans+1):train,:);
	
	% Doing linear regression to evaluate output weights, using reg parameter to allow having some error during training
	Wout = (X'*X + reg*eye(N))\(X'*U);   
	
%% plotting training phase

	figure(1)
	tvals = (trans+1):train;
	subplot(2,1,1)              % graph training signal versus reservoir output
	plot(tvals, U(:,coord), 'b', tvals, X*Wout(:,coord), 'r'); axis tight
	title('Training data (blue) and fitting (red)')
	subplot(2,1,2)              % graph normalized RMS error
	plot(tvals, sqrt(sum((X*Wout-U).^2,2)./(sum((X*Wout).^2,2)+sum(U.^2,2))))
	title('Error during training')
	axis tight
	%% Testing phase
	test = train+test_length;             % testing time
	
	U = Ute;
	
	XX = zeros(test, N);
	XX(1,:) = init;
	
	for n = 1:(test-1)      % predict using autonomous reservoir with feedback
	    XX(n+1,:) = (1-alpha) * XX(n,:) + alpha*tanh(XX(n,:)*A' + (XX(n,:)*Wout)*Win');
	end
	XX=XX*Wout;

	figure(2)
		h_fig=figure(2);
		tvals = 1:test;
		subplot(2,1,1)              % graph test signal versus reservoir prediction
		plot(tvals, U(:,coord), 'b', tvals, XX(:,coord), 'r'); axis tight
		title('Prediction (red) and experimental data (blue)')
		subplot(2,1,2)   % graph normalized RMS error
		plot(tvals, sqrt(sum((XX-U).^2,2)./(1)));
		name=['file=' filename '_moyenn=true_rho=' num2str(rho) '_alpha=' num2str(alpha) '_N=' num2str(N) '_coord=' num2str(coord) '_reg=' num2str(reg) '_seed=' num2str(seed) '.pdf']
		saveas(h_fig,name)

%%Plotter la TF des signaux prédits et réels pour comparer les spectres en puissance

	tf_U=fft(U);
	tf_XX=fft(XX);
	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Filtre butterworth (qui ne marche pas encore) pour avoir des spectres plus utilisables%%%%

%	WT = str2num(get(edit_spectra_filter,'String')) ; % recupere la valeur de coupure (sec) 
	% parametres filtrage passe-bas pour les spectres
	%WT = 2. ; % 'frequence' de coupure (en s)
	% frequence de coupure normalisee par le pas en frequence = 1/T
%	Wn = WT / (tfin-tdeb);

%	if (Wn > 1)
%		cprintf('Error','Impossible to filter spectrum for plateau %i at %10.1f s; reset to max value = %10.1f s\n',i_plat,WT,tfin-tdeb);

%			Wn = 0.99;
%	end

	% Filtre butterworth degre 2 ; frequence de coupure Wn
%	[num,denom] = butter(2,Wn,'low');

	% Filtrage aller-retour : zero-phase
%	log_power = filtfilt(num,denom,tf_U);

	
	%RefT = downsample(RefT,skip); % decrease sampling rate by handles.temperature.skip (i.e., 1000)
															
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	figure(3)
		h_fig_tf=figure(3);
		Fs=20;
		fs=(0:test-1)*(Fs/test);
		power_U=abs(tf_U).^2/test;
		power_XX=abs(tf_XX).^2/test;		
		semilogy(fs,power_U,'b',fs,power_XX,'r'); %on plot la TF en échelle log
		name=['file=' filename '_TF_moyenn=' num2str(moy) '_rho=' num2str(rho) '_alpha=' num2str(alpha) '_N=' num2str(N) '_coord=' num2str(coord) '_reg=' num2str(reg) '_seed=' num2str(seed) '.pdf'];
		saveas(h_fig_tf,name);
		
	[H,pValue,Lambda,Lambda_list,Order,CI]=chaostest(data(:,coord));
	[H_pred,pValue_pred,Lambda_pred,Lambda_list_pred,Order_pred,CI_pred]=chaostest(XX(:,coord));

	hor_lya=1/Lambda;
	hor_lya_pred=1/Lambda_pred;

end
