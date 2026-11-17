
clc
clear
close all
% Parâmetros iniciais
t0 = 0;
tf = 1;
h = 1;
t = t0:h:tf;
n = length(t);

% Função EDO: y' = f(t, y)
f = @(t, y) sin(y)+t;

% Solução exata para comparação
%y_exata = @(t)  1000*exp(0.04*t);

% Inicialização dos vetores
y_euler = zeros(1, n);
y_modif = zeros(1, n);

y_euler(1) = 1;
y_modif(1) = 1;

% Implementação do Método de Euler
for i = 1:n-1
    % Método de Euler Explícito
    f_euler = f(t(i), y_euler(i));
    y_euler(i+1) = y_euler(i) + h * f_euler;
    
end


% Implementação do Método de Euler Modificado
for i = 1:n-1
    % Método de Euler Explícito
    f_euler1 = f(t(i), y_modif(i));
    f_euler2 = f(t(i)+h, y_modif(i)+h*f_euler1);
    y_modif(i+1) = y_modif(i) + (h/2) * (f_euler1+f_euler2);
    
end

% Gráfico comparativo
figure;
plot(t, y_modif, 'k-', 'LineWidth', 2); hold on;
plot(t, y_euler, 'ro--', 'LineWidth', 1.5);
xlabel('Tempo (t)');
ylabel('Valor de y');
legend('Euler', 'Euler Modif', 'Location', 'Best');
title('Comparação: Método de Euler vs Método de Euler Modificado');
grid on;

u = y_modif
erro_absoluto=abs(y_euler - u)


for i = 1:n
   erro_percentual(i)=abs(erro_absoluto(i)/y_modif(i))*100;
end
erro_percentual


Erro_comparando_com_solucao_analitica = 2.412875-y_modif(n)

format long

A = [y_euler' y_modif' erro_percentual'];

T = array2table(A,VariableNames=["Euler" "Euler Modificado" "Erro Percentual"])

