import { Component } from '@angular/core';
import { Topbar } from '../topbar/topbar';

@Component({
  selector: 'app-dashboard-page',
  imports: [
    Topbar
  ],
  templateUrl: './dashboard-page.html',
  styleUrl: './dashboard-page.css',
})
export class DashboardPage {}
