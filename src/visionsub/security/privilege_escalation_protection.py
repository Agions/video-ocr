"""
权限提升防护模块
提供防止权限提升攻击的安全功能
"""

import os
import sys
import pwd
import grp
import subprocess
import logging
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import stat

logger = logging.getLogger(__name__)

class PrivilegeEscalationDetector:
    """权限提升检测器"""
    
    def __init__(self):
        self.privileged_operations = {
            'chown', 'chmod', 'mount', 'umount', 'mkfs', 'fdisk',
            'dd', 'kill', 'killall', 'pkill', 'skill', 'slay',
            'shutdown', 'reboot', 'halt', 'poweroff',
            'passwd', 'useradd', 'usermod', 'userdel',
            'groupadd', 'groupmod', 'groupdel',
            'sudo', 'su', 'doas', 'pkexec',
            'crontab', 'at', 'batch',
            'iptables', 'firewall-cmd', 'ufw',
            'systemctl', 'service', 'chkconfig',
            'insmod', 'rmmod', 'modprobe', 'lsmod',
            'dmesg', 'sysctl', 'proc',
            'setuid', 'setgid', 'setcap',
            'ptrace', 'gdb', 'strace', 'ltrace'
        }
        
        self.sensitive_files = {
            '/etc/passwd', '/etc/shadow', '/etc/sudoers',
            '/etc/hosts', '/etc/hostname', '/etc/resolv.conf',
            '/etc/ssh/sshd_config', '/etc/ssh/ssh_config',
            '/etc/hosts.allow', '/etc/hosts.deny',
            '/etc/crontab', '/etc/cron.d/', '/var/spool/cron/',
            '/etc/fstab', '/etc/mtab',
            '/boot/', '/lib/', '/lib64/', '/usr/lib/', '/usr/local/lib/',
            '/bin/', '/sbin/', '/usr/bin/', '/usr/sbin/', '/usr/local/bin/',
            '/etc/systemd/system/', '/lib/systemd/system/',
            '/var/log/', '/var/tmp/', '/tmp/',
            '/proc/', '/sys/', '/dev/'
        }
        
        self.privileged_ports = {1, 7, 9, 11, 13, 15, 17, 19, 20, 21, 22, 23, 25, 37, 42, 43, 53, 67, 68, 69, 70, 79, 80, 88, 101, 107, 109, 110, 111, 113, 115, 117, 119, 123, 135, 137, 138, 139, 143, 161, 162, 177, 179, 389, 427, 443, 444, 445, 464, 468, 500, 514, 515, 520, 521, 530, 543, 544, 545, 548, 554, 587, 631, 646, 873, 901, 993, 995, 1024, 1025, 1026, 1027, 1028, 1029, 1110, 1433, 1434, 1512, 1521, 2049, 2103, 2105, 2107, 3000, 3128, 3306, 3389, 5000, 5001, 5432, 5631, 5632, 5666, 5800, 5801, 5900, 6000, 6001, 6646, 6666, 6667, 6668, 6669, 6697, 6881, 6882, 6883, 6884, 6885, 6886, 6887, 6888, 6889, 6890, 6891, 6892, 6893, 6894, 6895, 6896, 6897, 6898, 6899, 6900, 6901, 6902, 6903, 6904, 6905, 6906, 6907, 6908, 6909, 6910, 6911, 6912, 6913, 6914, 6915, 6916, 6917, 6918, 6919, 6920, 6921, 6922, 6923, 6924, 6925, 6926, 6927, 6928, 6929, 6930, 6931, 6932, 6933, 6934, 6935, 6936, 6937, 6938, 6939, 6940, 6941, 6942, 6943, 6944, 6945, 6946, 6947, 6948, 6949, 6950, 6951, 6952, 6953, 6954, 6955, 6956, 6957, 6958, 6959, 6960, 6961, 6962, 6963, 6964, 6965, 6966, 6967, 6968, 6969, 6970, 6971, 6972, 6973, 6974, 6975, 6976, 6977, 6978, 6979, 6980, 6981, 6982, 6983, 6984, 6985, 6986, 6987, 6988, 6989, 6990, 6991, 6992, 6993, 6994, 6995, 6996, 6997, 6998, 6999, 7000, 7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010, 7011, 7012, 7013, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022, 7023, 7024, 7025, 7026, 7027, 7028, 7029, 7030, 7031, 7032, 7033, 7034, 7035, 7036, 7037, 7038, 7039, 7040, 7041, 7042, 7043, 7044, 7045, 7046, 7047, 7048, 7049, 7050, 7051, 7052, 7053, 7054, 7055, 7056, 7057, 7058, 7059, 7060, 7061, 7062, 7063, 7064, 7065, 7066, 7067, 7068, 7069, 7070, 7071, 7072, 7073, 7074, 7075, 7076, 7077, 7078, 7079, 7080, 7081, 7082, 7083, 7084, 7085, 7086, 7087, 7088, 7089, 7090, 7091, 7092, 7093, 7094, 7095, 7096, 7097, 7098, 7099, 7100, 7101, 7102, 7103, 7104, 7105, 7106, 7107, 7108, 7109, 7110, 7111, 7112, 7113, 7114, 7115, 7116, 7117, 7118, 7119, 7120, 7121, 7122, 7123, 7124, 7125, 7126, 7127, 7128, 7129, 7130, 7131, 7132, 7133, 7134, 7135, 7136, 7137, 7138, 7139, 7140, 7141, 7142, 7143, 7144, 7145, 7146, 7147, 7148, 7149, 7150, 7151, 7152, 7153, 7154, 7155, 7156, 7157, 7158, 7159, 7160, 7161, 7162, 7163, 7164, 7165, 7166, 7167, 7168, 7169, 7170, 7171, 7172, 7173, 7174, 7175, 7176, 7177, 7178, 7179, 7180, 7181, 7182, 7183, 7184, 7185, 7186, 7187, 7188, 7189, 7190, 7191, 7192, 7193, 7194, 7195, 7196, 7197, 7198, 7199, 7200, 7201, 7202, 7203, 7204, 7205, 7206, 7207, 7208, 7209, 7210, 7211, 7212, 7213, 7214, 7215, 7216, 7217, 7218, 7219, 7220, 7221, 7222, 7223, 7224, 7225, 7226, 7227, 7228, 7229, 7230, 7231, 7232, 7233, 7234, 7235, 7236, 7237, 7238, 7239, 7240, 7241, 7242, 7243, 7244, 7245, 7246, 7247, 7248, 7249, 7250, 7251, 7252, 7253, 7254, 7255, 7256, 7257, 7258, 7259, 7260, 7261, 7262, 7263, 7264, 7265, 7266, 7267, 7268, 7269, 7270, 7271, 7272, 7273, 7274, 7275, 7276, 7277, 7278, 7279, 7280, 7281, 7282, 7283, 7284, 7285, 7286, 7287, 7288, 7289, 7290, 7291, 7292, 7293, 7294, 7295, 7296, 7297, 7298, 7299, 7300, 7301, 7302, 7303, 7304, 7305, 7306, 7307, 7308, 7309, 7310, 7311, 7312, 7313, 7314, 7315, 7316, 7317, 7318, 7319, 7320, 7321, 7322, 7323, 7324, 7325, 7326, 7327, 7328, 7329, 7330, 7331, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7341, 7342, 7343, 7344, 7345, 7346, 7347, 7348, 7349, 7350, 7351, 7352, 7353, 7354, 7355, 7356, 7357, 7358, 7359, 7360, 7361, 7362, 7363, 7364, 7365, 7366, 7367, 7368, 7369, 7370, 7371, 7372, 7373, 7374, 7375, 7376, 7377, 7378, 7379, 7380, 7381, 7382, 7383, 7384, 7385, 7386, 7387, 7388, 7389, 7390, 7391, 7392, 7393, 7394, 7395, 7396, 7397, 7398, 7399, 7400, 7401, 7402, 7403, 7404, 7405, 7406, 7407, 7408, 7409, 7410, 7411, 7412, 7413, 7414, 7415, 7416, 7417, 7418, 7419, 7420, 7421, 7422, 7423, 7424, 7425, 7426, 7427, 7428, 7429, 7430, 7431, 7432, 7433, 7434, 7435, 7436, 7437, 7438, 7439, 7440, 7441, 7442, 7443, 7444, 7445, 7446, 7447, 7448, 7449, 7450, 7451, 7452, 7453, 7454, 7455, 7456, 7457, 7458, 7459, 7460, 7461, 7462, 7463, 7464, 7465, 7466, 7467, 7468, 7469, 7470, 7471, 7472, 7473, 7474, 7475, 7476, 7477, 7478, 7479, 7480, 7481, 7482, 7483, 7484, 7485, 7486, 7487, 7488, 7489, 7490, 7491, 7492, 7493, 7494, 7495, 7496, 7497, 7498, 7499, 7500, 7501, 7502, 7503, 7504, 7505, 7506, 7507, 7508, 7509, 7510, 7511, 7512, 7513, 7514, 7515, 7516, 7517, 7518, 7519, 7520, 7521, 7522, 7523, 7524, 7525, 7526, 7527, 7528, 7529, 7530, 7531, 7532, 7533, 7534, 7535, 7536, 7537, 7538, 7539, 7540, 7541, 7542, 7543, 7544, 7545, 7546, 7547, 7548, 7549, 7550, 7551, 7552, 7553, 7554, 7555, 7556, 7557, 7558, 7559, 7560, 7561, 7562, 7563, 7564, 7565, 7566, 7567, 7568, 7569, 7570, 7571, 7572, 7573, 7574, 7575, 7576, 7577, 7578, 7579, 7580, 7581, 7582, 7583, 7584, 7585, 7586, 7587, 7588, 7589, 7590, 7591, 7592, 7593, 7594, 7595, 7596, 7597, 7598, 7599, 7600, 7601, 7602, 7603, 7604, 7605, 7606, 7607, 7608, 7609, 7610, 7611, 7612, 7613, 7614, 7615, 7616, 7617, 7618, 7619, 7620, 7621, 7622, 7623, 7624, 7625, 7626, 7627, 7628, 7629, 7630, 7631, 7632, 7633, 7634, 7635, 7636, 7637, 7638, 7639, 7640, 7641, 7642, 7643, 7644, 7645, 7646, 7647, 7648, 7649, 7650, 7651, 7652, 7653, 7654, 7655, 7656, 7657, 7658, 7659, 7660, 7661, 7662, 7663, 7664, 7665, 7666, 7667, 7668, 7669, 7670, 7671, 7672, 7673, 7674, 7675, 7676, 7677, 7678, 7679, 7680, 7681, 7682, 7683, 7684, 7685, 7686, 7687, 7688, 7689, 7690, 7691, 7692, 7693, 7694, 7695, 7696, 7697, 7698, 7699, 7700, 7701, 7702, 7703, 7704, 7705, 7706, 7707, 7708, 7709, 7710, 7711, 7712, 7713, 7714, 7715, 7716, 7717, 7718, 7719, 7720, 7721, 7722, 7723, 7724, 7725, 7726, 7727, 7728, 7729, 7730, 7731, 7732, 7733, 7734, 7735, 7736, 7737, 7738, 7739, 7740, 7741, 7742, 7743, 7744, 7745, 7746, 7747, 7748, 7749, 7750, 7751, 7752, 7753, 7754, 7755, 7756, 7757, 7758, 7759, 7760, 7761, 7762, 7763, 7764, 7765, 7766, 7767, 7768, 7769, 7770, 7771, 7772, 7773, 7774, 7775, 7776, 7777, 7778, 7779, 7780, 7781, 7782, 7783, 7784, 7785, 7786, 7787, 7788, 7789, 7790, 7791, 7792, 7793, 7794, 7795, 7796, 7797, 7798, 7799, 7800, 7801, 7802, 7803, 7804, 7805, 7806, 7807, 7808, 7809, 7810, 7811, 7812, 7813, 7814, 7815, 7816, 7817, 7818, 7819, 7820, 7821, 7822, 7823, 7824, 7825, 7826, 7827, 7828, 7829, 7830, 7831, 7832, 7833, 7834, 7835, 7836, 7837, 7838, 7839, 7840, 7841, 7842, 7843, 7844, 7845, 7846, 7847, 7848, 7849, 7850, 7851, 7852, 7853, 7854, 7855, 7856, 7857, 7858, 7859, 7860, 7861, 7862, 7863, 7864, 7865, 7866, 7867, 7868, 7869, 7870, 7871, 7872, 7873, 7874, 7875, 7876, 7877, 7878, 7879, 7880, 7881, 7882, 7883, 7884, 7885, 7886, 7887, 7888, 7889, 7890, 7891, 7892, 7893, 7894, 7895, 7896, 7897, 7898, 7899, 7900, 7901, 7902, 7903, 7904, 7905, 7906, 7907, 7908, 7909, 7910, 7911, 7912, 7913, 7914, 7915, 7916, 7917, 7918, 7919, 7920, 7921, 7922, 7923, 7924, 7925, 7926, 7927, 7928, 7929, 7930, 7931, 7932, 7933, 7934, 7935, 7936, 7937, 7938, 7939, 7940, 7941, 7942, 7943, 7944, 7945, 7946, 7947, 7948, 7949, 7950, 7951, 7952, 7953, 7954, 7955, 7956, 7957, 7958, 7959, 7960, 7961, 7962, 7963, 7964, 7965, 7966, 7967, 7968, 7969, 7970, 7971, 7972, 7973, 7974, 7975, 7976, 7977, 7978, 7979, 7980, 7981, 7982, 7983, 7984, 7985, 7986, 7987, 7988, 7989, 7990, 7991, 7992, 7993, 7994, 7995, 7996, 7997, 7998, 7999, 8000}
        
        self.current_user = self._get_current_user()
        self.current_groups = self._get_current_groups()
    
    def _get_current_user(self) -> str:
        """获取当前用户"""
        try:
            return pwd.getpwuid(os.getuid()).pw_name
        except Exception:
            return str(os.getuid())
    
    def _get_current_groups(self) -> Set[str]:
        """获取当前用户组"""
        try:
            return {grp.getgrgid(gid).gr_name for gid in os.getgroups()}
        except Exception:
            return set()
    
    def is_root_user(self) -> bool:
        """检查是否为root用户"""
        return os.getuid() == 0
    
    def is_privileged_user(self) -> bool:
        """检查是否为特权用户"""
        privileged_groups = {
            'sudo', 'admin', 'wheel', 'root', 'adm', 'staff',
            'systemd-journal', 'docker', 'libvirt', 'kvm'
        }
        return any(group in privileged_groups for group in self.current_groups)
    
    def check_privileged_operations(self, command: str) -> List[str]:
        """检查命令中的特权操作"""
        issues = []
        
        # 检查特权命令
        for priv_op in self.privileged_operations:
            if priv_op in command:
                issues.append(f"检测到特权操作: {priv_op}")
        
        # 检查sudo使用
        if 'sudo' in command:
            issues.append("检测到sudo使用，可能存在权限提升风险")
        
        # 检查文件操作
        if any(sensitive_file in command for sensitive_file in self.sensitive_files):
            issues.append("检测到敏感文件操作")
        
        # 检查网络操作
        if any(str(port) in command for port in self.privileged_ports):
            issues.append("检测到特权端口操作")
        
        return issues
    
    def check_file_permissions(self, file_path: str) -> List[str]:
        """检查文件权限"""
        issues = []
        
        try:
            if not os.path.exists(file_path):
                return [f"文件不存在: {file_path}"]
            
            stat_info = os.stat(file_path)
            mode = stat_info.st_mode
            
            # 检查setuid位
            if mode & stat.S_ISUID:
                issues.append(f"文件设置了setuid位: {file_path}")
            
            # 检查setgid位
            if mode & stat.S_ISGID:
                issues.append(f"文件设置了setgid位: {file_path}")
            
            # 检查sticky位
            if mode & stat.S_ISVTX:
                issues.append(f"文件设置了sticky位: {file_path}")
            
            # 检查其他用户可写
            if mode & stat.S_IWOTH:
                issues.append(f"文件对其他用户可写: {file_path}")
            
            # 检查组用户可写
            if mode & stat.S_IWGRP:
                issues.append(f"文件对组用户可写: {file_path}")
            
            # 检查是否为可执行文件
            if mode & stat.S_IXUSR:
                # 检查是否为特权二进制文件
                if file_path.startswith('/bin/') or file_path.startswith('/sbin/') or file_path.startswith('/usr/bin/') or file_path.startswith('/usr/sbin/'):
                    issues.append(f"检测到系统可执行文件: {file_path}")
            
        except Exception as e:
            issues.append(f"检查文件权限失败: {e}")
        
        return issues
    
    def check_process_privileges(self) -> Dict[str, Any]:
        """检查进程权限"""
        result = {
            'is_root': self.is_root_user(),
            'is_privileged': self.is_privileged_user(),
            'current_user': self.current_user,
            'current_groups': list(self.current_groups),
            'effective_user': None,
            'effective_groups': [],
            'capabilities': [],
            'issues': []
        }
        
        try:
            # 获取有效用户ID
            result['effective_user'] = pwd.getpwuid(os.geteuid()).pw_name
            
            # 获取有效组
            result['effective_groups'] = [grp.getgrgid(gid).gr_name for gid in os.getgroups()]
            
            # 检查能力（Linux）
            if os.path.exists('/proc/self/status'):
                with open('/proc/self/status', 'r') as f:
                    for line in f:
                        if line.startswith('Cap'):
                            cap_name = line.split(':')[0].strip()
                            cap_value = line.split(':')[1].strip()
                            result['capabilities'].append(f"{cap_name}: {cap_value}")
            
            # 生成问题报告
            if result['is_root']:
                result['issues'].append("进程以root用户运行")
            
            if result['is_privileged']:
                result['issues'].append("进程具有特权组权限")
            
            if result['current_user'] != result['effective_user']:
                result['issues'].append(f"用户权限提升: {result['current_user']} -> {result['effective_user']}")
            
        except Exception as e:
            result['issues'].append(f"检查进程权限失败: {e}")
        
        return result
    
    def audit_sudo_usage(self) -> List[str]:
        """审计sudo使用情况"""
        issues = []
        
        try:
            # 检查sudoers文件
            sudoers_files = ['/etc/sudoers', '/etc/sudoers.d/']
            
            for sudoers_file in sudoers_files:
                if os.path.exists(sudoers_file):
                    if os.path.isfile(sudoers_file):
                        issues.extend(self._check_sudoers_file(sudoers_file))
                    elif os.path.isdir(sudoers_file):
                        for root, dirs, files in os.walk(sudoers_file):
                            for file in files:
                                file_path = os.path.join(root, file)
                                issues.extend(self._check_sudoers_file(file_path))
            
            # 检查sudo日志
            if os.path.exists('/var/log/auth.log') or os.path.exists('/var/log/secure'):
                issues.append("建议检查sudo使用日志")
            
        except Exception as e:
            issues.append(f"sudo审计失败: {e}")
        
        return issues
    
    def _check_sudoers_file(self, file_path: str) -> List[str]:
        """检查sudoers文件"""
        issues = []
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 检查危险配置
            dangerous_patterns = [
                r'ALL\s*=\s*\(ALL\)\s*ALL',
                r'ALL\s*=\s*\(ALL\)\s*NOPASSWD:',
                r'ALL\s*=\s*\(ALL\)\s*PASSWD:',
                r'.*\s*=\s*\(ALL\)\s*ALL',
                r'.*\s*=\s*\(ALL\)\s*NOPASSWD:',
            ]
            
            import re
            for pattern in dangerous_patterns:
                if re.search(pattern, content):
                    issues.append(f"检测到危险sudo配置: {pattern}")
            
            # 检查文件权限
            stat_info = os.stat(file_path)
            if stat_info.st_mode & stat.S_IWOTH:
                issues.append(f"sudoers文件对其他用户可写: {file_path}")
            
        except Exception as e:
            issues.append(f"检查sudoers文件失败: {file_path} - {e}")
        
        return issues
    
    def generate_security_report(self) -> str:
        """生成安全报告"""
        report = []
        report.append("=" * 60)
        report.append("VisionSub 权限提升安全报告")
        report.append("=" * 60)
        report.append("")
        
        # 当前用户信息
        report.append("当前用户信息:")
        report.append(f"  用户: {self.current_user}")
        report.append(f"  组: {', '.join(self.current_groups)}")
        report.append(f"  是否为root: {self.is_root_user()}")
        report.append(f"  是否为特权用户: {self.is_privileged_user()}")
        report.append("")
        
        # 进程权限
        process_info = self.check_process_privileges()
        report.append("进程权限:")
        report.append(f"  有效用户: {process_info['effective_user']}")
        report.append(f"  有效组: {', '.join(process_info['effective_groups'])}")
        
        if process_info['issues']:
            report.append("  问题:")
            for issue in process_info['issues']:
                report.append(f"    - {issue}")
        report.append("")
        
        # Sudo审计
        sudo_issues = self.audit_sudo_usage()
        if sudo_issues:
            report.append("Sudo配置问题:")
            for issue in sudo_issues:
                report.append(f"  - {issue}")
            report.append("")
        
        # 建议
        recommendations = self._generate_recommendations()
        if recommendations:
            report.append("安全建议:")
            for rec in recommendations:
                report.append(f"  - {rec}")
        
        return "\n".join(report)
    
    def _generate_recommendations(self) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if self.is_root_user():
            recommendations.append("建议以普通用户身份运行应用程序")
        
        if self.is_privileged_user():
            recommendations.append("检查是否需要特权组权限")
        
        # 检查系统文件权限
        system_files = [
            '/etc/passwd', '/etc/shadow', '/etc/sudoers',
            '/bin/sh', '/bin/bash', '/usr/bin/sudo'
        ]
        
        for file_path in system_files:
            if os.path.exists(file_path):
                issues = self.check_file_permissions(file_path)
                if issues:
                    recommendations.append(f"修复 {file_path} 的权限问题")
        
        recommendations.append("定期审计用户权限和组成员资格")
        recommendations.append("启用日志记录以监控权限使用情况")
        recommendations.append("使用最小权限原则配置应用程序")
        
        return recommendations

class PrivilegeManager:
    """权限管理器"""
    
    def __init__(self):
        self.detector = PrivilegeEscalationDetector()
        self.allowed_commands = set()
        self.restricted_paths = set()
    
    def add_allowed_command(self, command: str):
        """添加允许的命令"""
        self.allowed_commands.add(command)
    
    def add_restricted_path(self, path: str):
        """添加限制路径"""
        self.restricted_paths.add(path)
    
    def is_command_allowed(self, command: str) -> bool:
        """检查命令是否允许"""
        # 检查是否在允许列表中
        if command in self.allowed_commands:
            return True
        
        # 检查特权操作
        issues = self.detector.check_privileged_operations(command)
        if issues:
            logger.warning(f"命令包含特权操作: {command}")
            return False
        
        # 检查路径限制
        for restricted_path in self.restricted_paths:
            if restricted_path in command:
                logger.warning(f"命令访问限制路径: {command}")
                return False
        
        return True
    
    def drop_privileges(self):
        """放弃特权"""
        try:
            if not self.detector.is_root_user():
                return
            
            # 获取原始用户
            original_uid = os.getuid()
            original_gid = os.getgid()
            
            # 如果已经是普通用户，无需操作
            if original_uid != 0:
                return
            
            # 尝试切换到nobody用户
            try:
                import pwd
                nobody_info = pwd.getpwnam('nobody')
                nobody_uid = nobody_info.pw_uid
                nobody_gid = nobody_info.pw_gid
                
                # 设置组ID
                os.setgid(nobody_gid)
                
                # 设置用户ID
                os.setuid(nobody_uid)
                
                logger.info(f"权限已降低到nobody用户 (UID: {nobody_uid}, GID: {nobody_gid})")
                
            except Exception as e:
                logger.warning(f"无法切换到nobody用户: {e}")
        
        except Exception as e:
            logger.error(f"放弃特权失败: {e}")
    
    def create_sandbox(self) -> 'Sandbox':
        """创建沙箱环境"""
        return Sandbox(self)

class Sandbox:
    """沙箱环境"""
    
    def __init__(self, privilege_manager: PrivilegeManager):
        self.privilege_manager = privilege_manager
        self.original_environ = os.environ.copy()
        self.original_cwd = os.getcwd()
        self.original_umask = os.umask(0o077)
        
    def __enter__(self):
        """进入沙箱"""
        # 限制环境变量
        self._restrict_environment()
        
        # 改变工作目录到临时目录
        self._change_to_temp_dir()
        
        # 设置严格的umask
        os.umask(0o077)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出沙箱"""
        # 恢复环境变量
        os.environ.clear()
        os.environ.update(self.original_environ)
        
        # 恢复工作目录
        os.chdir(self.original_cwd)
        
        # 恢复umask
        os.umask(self.original_umask)
    
    def _restrict_environment(self):
        """限制环境变量"""
        # 只保留必要的环境变量
        safe_vars = {
            'HOME', 'USER', 'PATH', 'LANG', 'LC_ALL', 'TERM',
            'SHELL', 'LOGNAME', 'DISPLAY', 'XAUTHORITY'
        }
        
        # 移除危险的环境变量
        dangerous_vars = {
            'LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH',
            'PERL5LIB', 'RUBYLIB', 'CLASSPATH'
        }
        
        # 清理环境变量
        for var in list(os.environ.keys()):
            if var not in safe_vars:
                del os.environ[var]
        
        # 移除危险变量
        for var in dangerous_vars:
            if var in os.environ:
                del os.environ[var]
        
        # 设置安全的PATH
        os.environ['PATH'] = '/usr/bin:/bin'
        
        # 清理LD_PRELOAD
        os.environ.pop('LD_PRELOAD', None)
    
    def _change_to_temp_dir(self):
        """改变到临时目录"""
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix='visionsub_sandbox_')
        os.chdir(temp_dir)
        
        # 设置目录权限
        os.chmod(temp_dir, 0o700)

# 全局实例
privilege_detector = PrivilegeEscalationDetector()
privilege_manager = PrivilegeManager()