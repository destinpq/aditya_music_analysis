'use client';

import React from 'react';
import { ConfigProvider } from 'antd';

// Configure Ant Design theme to match your site's colors
const theme = {
  token: {
    colorPrimary: '#4338ca',
    colorBgBase: '#121212',
    colorTextBase: '#ffffff',
    colorBorder: '#333333',
    borderRadius: 8,
  },
};

export default function AntdLayout({ children }: { children: React.ReactNode }) {
  return (
    <ConfigProvider theme={theme}>
      {children}
    </ConfigProvider>
  );
} 