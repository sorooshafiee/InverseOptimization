function third_party()

    % Download YALMIP
    url = 'https://github.com/johanlofberg/YALMIP/archive/develop.zip';
    fprintf('Downloading YALMIP ... \n');
    websave('YALMIP.zip',url);
    unzip('YALMIP.zip','.');
    delete YALMIP.zip;
    
    % Download Sedumi
    url = 'https://github.com/sqlp/sedumi/archive/master.zip';
    fprintf('Downloading Sedumi ... \n');
    websave('Sedumi.zip',url);
    unzip('Sedumi.zip','.');
    delete Sedumi.zip;
end