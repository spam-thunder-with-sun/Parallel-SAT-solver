clear;
clc;
close all;

%Per i colori
global colors colorsCounter
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30", "#4DBEEE", "#A2142F"];
colorsCounter = 1; 

mathsat_data = readMathsatData("mathsat.txt");
myDPLL_data = readMyDPLLData("customDPLL.txt");
fclose('all');

myGraph(mathsat_data, myDPLL_data);

function myGraph(mathsat_data, myDPLL_data)
    mathsat_vars = mathsat_data(:, 1);
    mathsat_clauses = mathsat_data(:, 2);
    mathsat_sat = mathsat_data(:, 3)';
    mathsat_cpu = mathsat_data(:, 4)';
    %mathsat_mem = mathsat_data(:, 5)';

    myDPLL_vars = myDPLL_data(:, 1);
    myDPLL_clauses = myDPLL_data(:, 2);
    myDPLL_sat = myDPLL_data(:, 3)';
    myDPLL_cpu = myDPLL_data(:, 4)';
   
    %figure('Name', "SAT Solver Comparison")
    figure
    hold on;
    
    for i = 1:size(myDPLL_vars)

        p1 = plot3(mathsat_vars(i), mathsat_clauses(i), mathsat_cpu(i), 'o');    
        p1.Marker = '.';
        p1.MarkerSize = 30;

        if mathsat_sat(i) == 1
            p1.Color = "green";
        else
            p1.Color = "red";
        end
        
        p2 = plot3(myDPLL_vars(i), myDPLL_clauses(i), myDPLL_cpu(i), 'o'); 
        p2.Marker = 'o';
        p2.MarkerSize = 30;

        if myDPLL_sat(i) == 1
            p2.Color = "green";
        else
            p2.Color = "red";
        end

    end
    
    set(gca,'xscale','log');
    set(gca,'yscale','log');
    set(gca,'zscale','log');

    xlabel("n of literals");
    ylabel("n of clauses");
    zlabel("time of CPU ( s )");
    grid on
    title('SAT Solver Comparison');
    view(360, 0)%Letterali/tempo
    %view(90, 0)%Clausole/tempo
    %view(0, 90) %Test correttezza

    %view(45, 0) %Test correttezza

end

%-----------------------------------------------------------------------------------------

function data = readMyDPLLData(filename)
    myfile = fopen(filename, 'rt');

    %Aggiungo il padding per i dati più grandi
    
    data = [];

    while ~feof(myfile)
        %Salvo i dati del problema
        fscanf(myfile, "%s", 1);
        vars = fscanf(myfile, "%d", 1);
        fscanf(myfile, "%s", 1);
        clauses = fscanf(myfile, "%d", 1);
        fscanf(myfile, "%s", 1);
        %Salvo il risultato
        ris = fscanf(myfile, "%d", 1);
        %Salvo la cpi
        fscanf(myfile, "%s", 1);
        cpu = fscanf(myfile, "%f", 1);
        if cpu == 0
            cpu = 0.001;
        end
        fscanf(myfile, "%s", 1);
        data = [data; vars clauses ris cpu];
    end
end

function data = readMathsatData(filename)
    myfile = fopen(filename, 'rt');
    
    data = [];
    
    while ~feof(myfile)
        line = fgetl(myfile);

        %Leggo un nuovo problema
        if line == "c ============================================================================="            
            %Leggo i dati del problema
            fscanf(myfile, "%s", 3);
            vars = fscanf(myfile, "%d", 1);
            fscanf(myfile, "%s", 2);
            clauses = fscanf(myfile, "%d", 1);
            
            %Scarto il procedimento
            line = fgetl(myfile);
            while(line ~= "c =============================================================================")
                line = fgetl(myfile);
            end

            %Leggo il risultato
            fscanf(myfile, "%s", 1);
            ris = fscanf(myfile, "%s", 1);
            if ris == "UNSATISFIABLE"
                ris = 0;
            else
                ris = 1;
            end

            %Scarto 7 line
            for i = 1:8
                fgetl(myfile);
            end

            %Leggo il tempo cpu e la memoria
            fscanf(myfile, "%s", 3);
            cpu = fscanf(myfile, "%f ");
               
            if cpu == 0
                cpu = 0.001;
            end

            fscanf(myfile, "%s", 4);
            mem = fscanf(myfile, "%f", 1);

            data = [data; vars, clauses, ris, cpu, mem];
        end
    end
end
