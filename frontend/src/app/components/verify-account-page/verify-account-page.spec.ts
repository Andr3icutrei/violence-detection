import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountPage } from './verify-account-page';

describe('VerifyAccountPage', () => {
  let component: VerifyAccountPage;
  let fixture: ComponentFixture<VerifyAccountPage>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountPage],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountPage);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
