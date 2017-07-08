function str = value2latex(value)
    str = sprintf('%0.1e',value);
    [token, remain] = strtok(str,'e');
    remain = ['\times 10^{', num2str(str2double(remain(2:end))), '}'];
    str = [token, remain];
end