% =========================================================================
% RESOLUÇÃO DE PVC PELO MÉTODO DAS DIFERENÇAS FINITAS (MDF) COM GRÁFICO E TABELA
% =========================================================================
clear; clc; close all;

% --- Parâmetros do Problema ---
a = 1;          % Limite esquerdo
b = 2;          % Limite direito
alpha = 1;      % Condição de contorno em x=1: y(1) = 1
beta = 2;       % Condição de contorno em x=2: y(2) = 2

N = 10;         % Número de subintervalos (altere aqui se desejar)
h = (b - a)/N;  % Tamanho do passo
x = (a:h:b)';   % Vetor coluna dos nós x_i

% --- Funções da EDO no formato: y'' + p(x)y' + q(x)y = r(x) ---
p = @(x) 2./x;
q = @(x) -2./(x.^2);
r = @(x) sin(log(x))./(x.^2);

% --- Montagem do Sistema Linear A*Y = B ---
A = zeros(N-1, N-1);
B = zeros(N-1, 1);

for i = 1:N-1
    xi = x(i+1); 
    
    A_esq = 1/(h^2) - p(xi)/(2*h);   
    A_cen = -2/(h^2) + q(xi);         
    A_dir = 1/(h^2) + p(xi)/(2*h);   
    
    A(i, i) = A_cen;
    if i > 1, A(i, i-1) = A_esq; end
    if i < N-1, A(i, i+1) = A_dir; end
    
    B(i) = r(xi);
    
    if i == 1, B(i) = B(i) - A_esq * alpha; end
    if i == N-1, B(i) = B(i) - A_dir * beta; end
end

% --- Resolução e Construção do Vetor de Solução ---
Y_interno = A \ B;
Y_num = [alpha; Y_interno; beta];

% --- Solução Exata Analítica ---
c1 = (69 + 12*sin(log(2)) + 4*cos(log(2))) / 70;
c2 = 11/10 - c1;
y_exata_fun = @(x) c1*x + c2./(x.^2) - 0.3*sin(log(x)) - 0.1*cos(log(x));

% Avalia a solução exata nos mesmos pontos x para a tabela
Y_exata = y_exata_fun(x); 

% Linha fina/suave contínua exclusiva para o traçado do gráfico exato
x_fino = linspace(a, b, 200)'; 
Y_exata_fino = y_exata_fun(x_fino);

% --- Geração da Tabela de Dados ---
erro_absoluto = abs(Y_num - Y_exata);
ResultadoTabela = table(x, Y_num, Y_exata, erro_absoluto, ...
    'VariableNames', {'x_i', 'Y_Numerico', 'Y_Exato', 'Erro_Absoluto'});

fprintf('=============================================================\n');
fprintf('     RESULTADOS DO MÉTODO DAS DIFERENÇAS FINITAS (N = %d)    \n', N);
fprintf('=============================================================\n');
disp(ResultadoTabela);

% --- Plotagem do Gráfico Comparativo ---
figure('Color', [1 1 1]);
% Plota a linha analítica contínua e suave
plot(x_fino, Y_exata_fino, 'b-', 'LineWidth', 2); hold on;
% Plota os pontos calculados numericamente por Diferenças Finitas
plot(x, Y_num, 'ro--', 'MarkerFaceColor', 'r', 'MarkerSize', 6, 'LineWidth', 1.2);

% Configurações visuais do gráfico
grid on;
set(gca, 'GridLineStyle', ':', 'GridAlpha', 0.6);
xlabel('Eixo x', 'FontSize', 11);
ylabel('Solução y(x)', 'FontSize', 11);
title('Comparação Digital: Solução Numérica (MDF) vs. Analítica Exata', 'FontSize', 12);
legend('Solução Exata (Analítica)', ['Aproximação MDF (N = ' num2str(N) ')'], ...
       'Location', 'NorthWest', 'FontSize', 10);
