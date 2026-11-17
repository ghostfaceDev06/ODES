
clc
clear
close all
% Parâmetros iniciais
t0 = 0;
tf = 1;
h = 0.1;
t = t0:h:tf;
n = length(t);

% Função EDO: y' = f(t, y)
f = @(t, y) 0.04*y;

% Solução exata para comparação
y_exata = @(t)  1000*exp(0.04*t);

% Inicialização dos vetores
y_euler = zeros(1, n);
y_heun = zeros(1, n);

y_euler(1) = 1000;

% Implementação do Método de Euler Modificado
for i = 1:n-1
    % Método de Euler Explícito
    f_euler1 = f(t(i), y_euler(i));
    f_euler2 = f(t(i)+h, y_euler(i)+h*f_euler1);
    y_euler(i+1) = y_euler(i) + (h/2) * (f_euler1+f_euler2);
    
end

% Gráfico comparativo
figure;
plot(t, y_exata(t), 'k-', 'LineWidth', 2); hold on;
plot(t, y_euler, 'ro--', 'LineWidth', 1.5);
xlabel('Tempo (t)');
ylabel('Valor de y');
legend('Exata', 'Euler Modif', 'Location', 'Best');
title('Comparação: Método de Euler vs Método de Heun');
grid on;

u = y_exata(t)
erro_absoluto=abs(y_euler - u)


for i = 1:n
   erro_percentual(i)=abs(erro_absoluto(i)/y_euler(i))*100;
end
erro_percentual

format long

A = [y_euler' u' erro_percentual'];

T = array2table(A,VariableNames=["Euler Modif" "Exata" "Erro Percentual"])

