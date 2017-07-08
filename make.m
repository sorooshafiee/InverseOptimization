function make(CPLEX_folder)
    try
        if isunix || ismac
            files = dir('./cpp/*.cpp');
            if nargin == 0
                CPLEX_folder = '/usr/pack/cplex-12.6-mr/amd64-debian-linux7.1/CPLEX_Studio';
            end
            cd ./model/
            for filename = {files.name} 
                fprintf('Mexing %s ... \n',filename{1});
                str1 = ['-I"', CPLEX_folder, '/concert/include" '];
                str2 = ['-I"', CPLEX_folder, '/cplex/include" '];
                str3 = ['-L"', CPLEX_folder, '/cplex/lib/x86-64_linux/static_pic" '];
                str4 = ['-L"', CPLEX_folder, '/concert/lib/x86-64_linux/static_pic" '];
                str5 = '-lilocplex -lconcert -lcplex -lm -lpthread ../cpp/';
                str = ['mex ', str1, str2, str3, str4, str5, filename{1},';'];
                eval(str);
            end
            cd ..
        elseif ispc
            files = dir('.\cpp\*.cpp');
            if nargin == 0
                CPLEX_folder = 'C:\IBM\ILOG\CPLEX_Studio126';
            end
            cd ./model/
            for filename = {files.name} 
                fprintf('Mexing %s ... \n',filename{1});
                str1 = ['-I"', CPLEX_folder, '\concert\include" '];
                str2 = ['-I"', CPLEX_folder, '\cplex\include" '];
                str3 = ['-L"', CPLEX_folder, '\cplex\lib\x64_windows_vs2012\stat_mda" '];
                str4 = ['-L"', CPLEX_folder, '\concert\lib\x64_windows_vs2012\stat_mda" '];
                str5 = '-lilocplex -lconcert -lcplex1260 ../cpp/';
                str = ['mex ', str1, str2, str3, str4, str5, filename{1},';'];
                eval(str);
            end
            cd ..
        end            
    catch ME
        rethrow(ME)
    end
end